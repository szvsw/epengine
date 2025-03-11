"""Submit a GIS job to a Hatchet workflow."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

import boto3
import geopandas as gpd
import yaml
from epinterface.sbem.fields.spec import SemanticModelFields
from pydantic import BaseModel, Field

from epengine.gis.data.epw_metadata import closest_epw
from epengine.gis.geometry import (
    convert_neighbors,
    inject_neighbor_ixs,
    inject_rotated_rectangles,
)
from epengine.models.leafs import AvailableWorkflowSpecs, WorkflowName

logger = logging.getLogger(__name__)


class GisJobAssumptions(BaseModel):
    """The assumptions for a GIS job to handle missing data."""

    wwr_ratio: float = Field(
        default=0.5, description="The window-to-wall ratio for the building."
    )
    num_floors: int = Field(
        default=2, description="The number of floors for the building."
    )
    f2f_height: float = Field(
        default=3.5, description="The height of the floor-to-floor height."
    )


class GisJobArgs(BaseModel):
    """The configuration for a GIS job."""

    gis_file: str = Field(..., description="The path to the GIS file.")
    db_file: str = Field(..., description="The path to the db file.")
    component_map: str = Field(..., description="The path to the component map.")
    semantic_fields: str = Field(..., description="The path to the semantic fields.")
    experiment_id: str = Field(..., description="The id of the experiment.")
    cart_crs: str = Field(
        ..., description="The crs of the cartesian coordinate system to project to."
    )
    leaf_workflow: WorkflowName = Field(..., description="The workflow to use.")
    bucket: str = Field(default="ml-for-bem", description="The bucket to use.")
    bucket_prefix: str = Field(
        default="hatchet", description="The prefix of the bucket."
    )
    existing_artifacts: Literal["overwrite", "forbid"] = Field(
        default="forbid", description="Whether to overwrite existing artifacts."
    )
    epw_query: str | None = Field(
        default="source in ['tmyx']",
        description="The pandas df query to use for the epw (e.g. to only return tmyx)",
    )
    recursion_factor: int = Field(
        default=10, description="The recursion factor for scatter/gather subdivision"
    )
    max_depth: int = Field(
        default=2, description="The max depth for scatter/gather subdivision."
    )
    assumptions: GisJobAssumptions = Field(
        default_factory=GisJobAssumptions,
        description="The assumptions for the GIS job.",
    )


def submit_gis_job(  # noqa: C901
    config: GisJobArgs,
    log_fn: Callable | None = None,
):
    """Convert a GIS file to simulation specifications.

    inputs:
        - gis_file: str
        - db_file: str
        - component_map: str
        - experiment_id: str

        [meta cols?]
        - height?
        - wwr?

        - bucket: str
        - bucket_prefix: str
        - existing_artifacts: Literal["overwrite", "forbid"]
        - recursion_factor: int
        - max_depth: int

    steps:
        - upload gis file
        - upload db file
        - upload component map
        [
        - open gis file
        - project gis file to cartesian crs
        - fit rotated rectangles around buildings
        - extract relevant geometric properties (footprint area / rotated rectangle area, long edge angle, short edge length, long edge length, aspect ratio)
        - add relevant fields like sort index.
        - closest epw for each building
        - compute neighbors
        - extract and store neighbors (footprints, heights) as wkt
        - create model specs df (convert polys to wkt)
        - upload model specs df
        - check component map and model workflow compatibility
        - check leaf spec and gdf compatibility?
        - create scatter gather job payload
        - submit job
        ]

    Args:
        config (GisJobArgs): The configuration for the job.
        log_fn (Callable | None): The function to use for logging.

    """
    log = log_fn or logger.info
    gis_file = Path(config.gis_file)
    db_file = Path(config.db_file)
    component_map = Path(config.component_map)
    _semantic_fields = Path(config.semantic_fields)
    experiment_id = config.experiment_id
    cart_crs = config.cart_crs
    leaf_workflow = config.leaf_workflow
    bucket = config.bucket
    bucket_prefix = config.bucket_prefix
    existing_artifacts = config.existing_artifacts
    epw_query = config.epw_query
    _recursion_factor = config.recursion_factor
    _max_depth = config.max_depth

    # TODO: trigger hatchet job for gis processing

    # open gis file
    gdf = cast(gpd.GeoDataFrame, gpd.read_file(gis_file))

    if not gdf.crs:
        msg = "GIS file has no crs.  Please set the CRS before running this script."
        raise ValueError(msg)

    if gdf.crs == "EPSG:3857":
        gdf = cast(gpd.GeoDataFrame, gdf.to_crs("EPSG:4326"))
    current_crs = gdf.crs

    log(f"GIS file has crs {current_crs}")
    if current_crs not in ["EPSG:4326", cart_crs]:
        msg = f"GIS file has crs {current_crs}.  Please set the CRS to 'EPSG:4326' or '{cart_crs}' before running this script."
        raise ValueError(msg)

    if current_crs != "EPSG:4326":
        log("Projecting gis file to EPSG:4326.")
        gdf = cast(gpd.GeoDataFrame, gdf.to_crs("EPSG:4326"))

    # load the semantic fields
    with open(_semantic_fields) as f:
        semantic_fields = SemanticModelFields.model_validate(yaml.safe_load(f))

    # project gis file to cartesian crs
    # and inject rotated rectangles and neighbor indices

    gdf, injected_geo_cols = inject_rotated_rectangles(gdf, cart_crs)
    gdf, injected_ix_cols = inject_neighbor_ixs(gdf)

    has_floor_col = semantic_fields.Num_Floors_col is not None
    has_height_col = semantic_fields.Height_col is not None
    if not semantic_fields.Num_Floors_col and not semantic_fields.Height_col:
        msg = "No floor or height column found in semantic fields."
        raise ValueError(msg)
    if has_floor_col and not has_height_col:
        semantic_fields.Height_col = "IMPUTED_HEIGHT"
        gdf[semantic_fields.Height_col] = (
            config.assumptions.f2f_height * gdf[semantic_fields.Num_Floors_col]
        )
    elif not has_floor_col and has_height_col:
        semantic_fields.Num_Floors_col = "IMPUTED_NUM_FLOORS"
        gdf[semantic_fields.Num_Floors_col] = (
            (gdf[semantic_fields.Height_col] // config.assumptions.f2f_height)
            .clip(1, None)
            .astype(int)
        )

    # TODO: add checks for crazy data

    gdf, injected_neighbor_cols = convert_neighbors(
        gdf,
        neighbor_col="neighbor_ixs",
        geometry_col="rotated_rectangle",
        neighbor_geo_out_col="neighbor_polys",
        num_floors_col=semantic_fields.Num_Floors_col,  # pyright: ignore [reportArgumentType]
        neighbor_floors_out_col="neighbor_floors",
        fill_na_val=config.assumptions.num_floors,
    )

    # create model specs df

    epw_meta = closest_epw(
        cast(gpd.GeoSeries, gdf["rotated_rectangle"].centroid),
        source_filter=epw_query,
        crs=cart_crs,
        log_fn=log,
    )
    gdf["epwzip_path"] = epw_meta["path"].apply(
        lambda x: Path(x).as_posix().split("onebuilding/")[-1]
    )
    for _ix, row in gdf.iterrows():
        for key, val in row.items():
            log(f"{key}: {val}")

    _workflow = AvailableWorkflowSpecs[leaf_workflow]

    s3 = boto3.client("s3")
    remote_root = f"{bucket_prefix}/{experiment_id}"

    def format_s3_key(folder, key):
        return f"{remote_root}/{folder}/{key}"

    def format_s3_uri(folder, key):
        return f"s3://{bucket}/{format_s3_key(folder, key)}"

    if False:
        # check if experiment_id already exists
        if (
            s3.list_objects_v2(Bucket=bucket, Prefix=remote_root).get("KeyCount", 0) > 0
            and existing_artifacts == "forbid"
        ):
            msg = f"Experiment '{experiment_id}' already exists. Set 'existing_artifacts' to 'overwrite' to confirm."
            raise ValueError(msg)

        # upload gis file
        gis_key = format_s3_key("artifacts", gis_file.name)
        s3.upload_file(gis_file.as_posix(), bucket, gis_key)

        # upload db file
        db_key = format_s3_key("artifacts", db_file.name)
        s3.upload_file(db_file.as_posix(), bucket, db_key)

        # upload component map
        component_map_key = format_s3_key("artifacts", component_map.name)
        s3.upload_file(component_map.as_posix(), bucket, component_map_key)
