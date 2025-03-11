"""Submit a GIS job to a Hatchet workflow."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

import boto3
import geopandas as gpd

from epengine.gis.data.epw_metadata import closest_epw
from epengine.gis.geometry import (
    convert_neighbors,
    inject_neighbor_ixs,
    inject_rotated_rectangles,
)
from epengine.models.leafs import AvailableWorkflowSpecs, WorkflowName

logger = logging.getLogger(__name__)


def submit_gis_job(
    gis_file: Path,
    db_file: Path,
    component_map: Path,
    experiment_id: str,
    cart_crs: str,
    leaf_workflow: WorkflowName,
    bucket: str = "ml-for-bem",
    bucket_prefix: str = "hatchet",
    existing_artifacts: Literal["overwrite", "forbid"] = "forbid",
    epw_query: str | None = "source in ['tmyx']",
    recursion_factor: int = 10,
    max_depth: int = 2,
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
        gis_file (Path): The path to the GIS file.
        db_file (Path): The path to the db file.
        component_map (Path): The path to the component map.
        experiment_id (str): The id of the experiment.
        cart_crs (str): The crs of the cartesian coordinate system to project to.
        leaf_workflow (WorkflowName): The workflow to use.
        bucket (str): The bucket to use.
        bucket_prefix (str): The prefix of the bucket.
        existing_artifacts (Literal["overwrite", "forbid"]): Whether to overwrite existing artifacts.
        epw_query (str | None): The pandas df query to use for the epw (e.g. to only return tmyx)
        recursion_factor (int): The recursion factor for scatter/gather subdivision
        max_depth (int): The max depth for scatter/gather subdivision.
        log_fn (Callable | None): The function to use for logging.

    """
    log = log_fn or logger.info

    # TODO: trigger hatchet job for gis processing

    # open gis file
    gdf = cast(gpd.GeoDataFrame, gpd.read_file(gis_file))

    if not gdf.crs:
        msg = "GIS file has no crs.  Please set the CRS before running this script."
        raise ValueError(msg)
    current_crs = gdf.crs
    if current_crs not in ["EPSG:4326", cart_crs]:
        msg = f"GIS file has crs {current_crs}.  Please set the CRS to 'EPSG:4326' or '{cart_crs}' before running this script."
        raise ValueError(msg)

    if current_crs != "EPSG:4326":
        log("Projecting gis file to EPSG:4326.")
        gdf = cast(gpd.GeoDataFrame, gdf.to_crs("EPSG:4326"))

    # project gis file to cartesian crs
    # and inject rotated rectangles and neighbor indices
    gdf, injected_geo_cols = inject_rotated_rectangles(gdf, cart_crs)
    gdf, injected_ix_cols = inject_neighbor_ixs(gdf)
    gdf, injected_neighbor_cols = convert_neighbors(
        gdf,
        neighbor_col="neighbor_ixs",
        geometry_col="rotated_rectangle",
        neighbor_geo_out_col="neighbor_polys",
        num_floors_col="height",
        neighbor_floors_out_col="neighbor_heights",
        fill_na_val=8,
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
