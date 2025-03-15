"""Submit a GIS job to a Hatchet workflow."""

import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

import boto3
import geopandas as gpd
import pandas as pd
import yaml
from epinterface.sbem.fields.spec import CategoricalFieldSpec, SemanticModelFields
from hatchet_sdk import new_client
from shapely import to_wkt

from epengine.gis.data.epw_metadata import closest_epw
from epengine.gis.geometry import (
    convert_neighbors,
    inject_neighbor_ixs,
    inject_rotated_rectangles,
)
from epengine.gis.models import GisJobArgs
from epengine.models.leafs import AvailableWorkflowSpecs

logger = logging.getLogger(__name__)
# TODO: this client should be imported
client = new_client()


def submit_gis_job(  # noqa: C901
    config: GisJobArgs,
    log_fn: Callable | None = None,
):
    """Convert a GIS file to simulation specifications.

    steps:
        TODO: this is slightly out of date and should be updated
        once this method finishes.
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

    Returns:
        dict[str, Any]: The response from the hatchet job.
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
    if leaf_workflow != "simulate_sbem_mapped":
        msg = f"Leaf workflow {leaf_workflow} is not supported; currently only 'simulate_sbem_mapped' is supported."
        raise ValueError(msg)

    # TODO: trigger hatchet job for gis processing

    gdf = cast(gpd.GeoDataFrame, gpd.read_file(gis_file))

    # Handle GIS transformations.
    # We will want to make sure that the GIS file is in either 4326, 3857,
    # or the expected cartesian reference.  If it's not, we will raise an
    # error to be safe for no mismatched expectations.
    # Then we will project to 4326 so we can safely infer lat lons and so on.
    log("Checking GIS file crs and reprojecting if necessary...")
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
    # We will need to access this as it stores some
    # rich information about which columns in the provided GIS data will
    # contain standard provided values, like wwr, height etc etc.
    # it also stores the fields that will be used for semantic mapping,
    # so we can run a consistency check with the component map.
    # TODO: this could become a cached_property of the config
    with open(_semantic_fields) as f:
        semantic_fields = SemanticModelFields.model_validate(yaml.safe_load(f))

    # We need to deal with the fact that shapefiles will trucnate the column
    # name to 10 characters, but users might not realize this when they
    # export from e.g. ArcGIS.
    if gis_file.suffix.lower() in [".zip", ".shp", ".shx"]:
        # Deal with truncated titles
        for field in semantic_fields.Fields:
            if field.Name[:10] in gdf.columns:
                log(
                    f"Renaming column '{field.Name[:10]}' to '{field.Name}' as per manifest."
                )
                gdf = cast(
                    gpd.GeoDataFrame, gdf.rename(columns={field.Name[:10]: field.Name})
                )

        # TODO: adding new rich semantic field columns to
        # to the model would require knowing to update this
        # list; this logic should be refactored out to exist on semantic fields.
        for col_name in [
            semantic_fields.Height_col,
            semantic_fields.Num_Floors_col,
            semantic_fields.WWR_col,
            semantic_fields.GFA_col,
        ]:
            if col_name is None:
                continue
            if col_name[:10] in gdf.columns:
                log(
                    f"Renaming column '{col_name[:10]}' to '{col_name}' as per manifest."
                )
                gdf = cast(
                    gpd.GeoDataFrame, gdf.rename(columns={col_name[:10]: col_name})
                )

    # We want to run a consistency check to make sure that the requested semantic fields
    # are actually in the GDF after we have dealt with appropriate renaming.
    # We also should run a consistency check to make sure that every cell value that is listed as an optional
    # semantic field is actually one of the expected values.
    for field in semantic_fields.Fields:
        if field.Name not in gdf.columns:
            msg = f"Field '{field.Name}' not found in gdf. Available columns: {gdf.columns.tolist()}"
            raise ValueError(msg)
        if isinstance(field, CategoricalFieldSpec):
            # TODO: this should become a method of the field spec,
            # which ought to have an ABC method for checking consistency
            # AND handling optional fallback behavior for na values.
            field_vals = gdf[field.Name]
            is_valid = field_vals.isin(field.Options)
            is_not_valid = ~is_valid
            if cast(pd.Series, is_not_valid).any():
                msg = f"Field '{field.Name}' has values which are not in the allowed options: {field.Options}"
                raise ValueError(msg)
        else:
            msg = f"Field '{field.Name}' is a  {field.__class__.__name__}, which does not yet have a consistency rule enabled."
            raise NotImplementedError(msg)

    # First, we will inject a fitted rotated rectangle in cartesian space
    # along with various geometric feature extractions (e.g. the length of the long edge,
    # the orientation of the long edge, and so on.)
    log("Performing geometry processing...")
    gdf, injected_geo_cols = inject_rotated_rectangles(gdf, cart_crs)

    # Then we will determine the neighbor indices using a culling radius of 100m.
    gdf, injected_ix_cols = inject_neighbor_ixs(gdf, neighbor_threshold=100, log_fn=log)

    # TODO: this should be a validator on the semantic fields object,
    # not in the gis submission workflow.
    has_floor_col = semantic_fields.Num_Floors_col is not None
    has_height_col = semantic_fields.Height_col is not None
    if not semantic_fields.Num_Floors_col and not semantic_fields.Height_col:
        msg = "No floor or height column found in semantic fields."
        raise ValueError(msg)

    # TODO: this could become a method of the config which accepts a dataframe
    # and returns a series
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
    if semantic_fields.Height_col is None or semantic_fields.Num_Floors_col is None:
        msg = "There was an issue determining the height or number of floors for the building."
        raise ValueError(msg)

    # TODO: deal with fractional floor nums?

    # # TODO: gfa computation/injection?
    # if semantic_fields.GFA_col is None:
    #     semantic_fields.GFA_col = "IMPUTED_GFA"
    #     gdf[semantic_fields.GFA_col] = (
    #         gdf.rotated_rectangle.area * gdf[semantic_fields.Num_Floors_col]
    #     )
    if semantic_fields.WWR_col is None:
        semantic_fields.WWR_col = "IMPUTED_WWR"
        gdf[semantic_fields.WWR_col] = config.assumptions.wwr_ratio
        log("WWR column was imputed.")
    else:
        if semantic_fields.WWR_col not in gdf.columns:
            msg = f"WWR column '{semantic_fields.WWR_col}' not found in gdf."
            raise ValueError(msg)
        missing_wwr = cast(pd.Series, gdf[semantic_fields.WWR_col].isna())
        gdf.loc[missing_wwr, semantic_fields.WWR_col] = config.assumptions.wwr_ratio
        if missing_wwr.any():
            msg = "Some WWR values were imputed."
            log(msg)
        if (
            (gdf[semantic_fields.WWR_col] < 0) | (gdf[semantic_fields.WWR_col] > 1)
        ).any():
            msg = "Some WWR values were out of range (0, 1)."
            log(msg)

    # TODO: add checks for crazy data on floor area, footprint area, etc.

    neighbor_geo_col = "neighbor_polys"
    neighbor_heights_col = "neighbor_heights"
    neighbor_floors_col = "neighbor_floors"
    gdf, injected_neighbor_cols = convert_neighbors(
        gdf,
        neighbor_col="neighbor_ixs",
        geometry_col="rotated_rectangle",
        neighbor_geo_out_col=neighbor_geo_col,
        height_col=semantic_fields.Height_col,  # pyright: ignore [reportArgumentType]
        neighbor_heights_out_col=neighbor_heights_col,
        fill_na_val=config.assumptions.num_floors * config.assumptions.f2f_height,
    )
    gdf[neighbor_floors_col] = gdf[neighbor_heights_col].apply(
        lambda x: [
            (v // config.assumptions.f2f_height) if v is not None else None for v in x
        ]
    )

    epw_meta = closest_epw(
        cast(gpd.GeoSeries, gdf["rotated_rectangle"].centroid),
        source_filter=epw_query,
        crs=cart_crs,
        log_fn=log,
    )

    def handle_epw_path(x: str):
        x = "/".join(x.split("\\")[2:-1])
        x = f"https://climate.onebuilding.org/{x}.zip"
        return x

    gdf["epwzip_uri"] = epw_meta["path"].apply(handle_epw_path)

    log("Uploading artifacts to s3...")
    s3 = boto3.client("s3")
    remote_root = f"{bucket_prefix}/{experiment_id}"

    def format_s3_key(folder, key):
        return f"{remote_root}/{folder}/{key}"

    def format_s3_uri(folder, key):
        return f"s3://{bucket}/{format_s3_key(folder, key)}"

    # check if experiment_id already exists
    if (
        s3.list_objects_v2(Bucket=bucket, Prefix=remote_root).get("KeyCount", 0) > 0
        and existing_artifacts == "forbid"
    ):
        msg = f"Experiment '{experiment_id}' already exists. Set 'existing_artifacts' to 'overwrite' to confirm."
        raise ValueError(msg)

    # upload gis file
    gis_key = format_s3_key("artifacts", gis_file.name)
    _gis_uri = format_s3_uri("artifacts", gis_file.name)
    s3.upload_file(gis_file.as_posix(), bucket, gis_key)

    # upload db file
    db_key = format_s3_key("artifacts", db_file.name)
    db_uri = format_s3_uri("artifacts", db_file.name)
    s3.upload_file(db_file.as_posix(), bucket, db_key)

    # upload component map
    component_map_key = format_s3_key("artifacts", component_map.name)
    component_map_uri = format_s3_uri("artifacts", component_map.name)
    s3.upload_file(component_map.as_posix(), bucket, component_map_key)

    # upload semantic fields; we need to
    # update it since some of the height/wwr/etc cols have been imputed.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / f"{_semantic_fields.stem}_updated.yml"
        with open(temp_path, "w") as f:
            yaml.dump(semantic_fields.model_dump(mode="json"), f, indent=2)
        semantic_fields_key = format_s3_key("artifacts", temp_path.name)
        semantic_fields_uri = format_s3_uri("artifacts", temp_path.name)
        s3.upload_file(temp_path.as_posix(), bucket, semantic_fields_key)

    # Now we must serialize the gdf into a parquet file.
    # TODO: a for loop is obviously a bad choice here.
    # TODO: this should probably be done with columnar access + renaming.
    # e.g. gdf[desired_cols].rename(columns={old: new, old: new})
    # However, we want to do some model validation here.
    # could probably do the vectorized ops and then the model validation
    # with a single apply call.
    all_data = []
    workflow_spec = AvailableWorkflowSpecs[leaf_workflow]
    for i, (_ix, row) in enumerate(gdf.iterrows()):
        data = {
            "experiment_id": experiment_id,
            "sort_index": i,
            "db_uri": db_uri,
            "semantic_fields_uri": semantic_fields_uri,
            "component_map_uri": component_map_uri,
            "epwzip_uri": row["epwzip_uri"],
            # TODO: this might be greatly increasing the size of the pq
            # file because it has to encode a textual column which is json
            # it might be more natural to instead leave them as their own columns
            # and then just load those columns after opening the yaml file in the
            # leaf task to determine the fields by reference.
            # however, doing so would require a two step process involving a
            # dynamic create_model call and two step validation.
            "semantic_field_context": {
                field.Name: row[field.Name] for field in semantic_fields.Fields
            },
            "neighbor_polys": [to_wkt(poly) for poly in row[neighbor_geo_col]],
            "neighbor_heights": row[neighbor_heights_col],
            "neighbor_floors": row[neighbor_floors_col],
            "rotated_rectangle": to_wkt(row["rotated_rectangle"]),
            # TODO: consider packaging up the injected columns into to avoid missing any
            # if the list changes.
            # so that they can be dynamically added to the dict
            "long_edge_angle": row["long_edge_angle"],
            "long_edge": row["long_edge"],
            "short_edge": row["short_edge"],
            "aspect_ratio": row["aspect_ratio"],
            "rotated_rectangle_area_ratio": row["rotated_rectangle_area_ratio"],
            "wwr": row[semantic_fields.WWR_col],
            "height": row[semantic_fields.Height_col],
            "num_floors": row[semantic_fields.Num_Floors_col],
            "f2f_height": config.assumptions.f2f_height,  # TODO: bring this into rich fields.
            # TODO: areas.
        }
        workflow_spec.model_validate(data)
        all_data.append(data)

    model_specs_df = pd.DataFrame(all_data)

    # upload model specs df
    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = Path(temp_dir) / "specs.pq"
        model_specs_key = format_s3_key("artifacts", local_path.name)
        model_specs_uri = format_s3_uri("artifacts", local_path.name)
        model_specs_df.to_parquet(local_path)
        s3.upload_file(local_path.as_posix(), bucket, model_specs_key)

    log("Submitting job to hatchet....")

    # TODO: use some validation here rather than dicts
    job_payload = {
        "experiment_id": experiment_id,
        "specs": model_specs_uri,
        "bucket": bucket,
        "recursion_map": {
            "factor": _recursion_factor,
            "max_depth": _max_depth,
        },
        "workflow_name": leaf_workflow,
    }
    # TODO: use a proper enum for the workflow name.
    workflowRef = client.admin.run_workflow(
        workflow_name="scatter_gather_recursive",
        input=job_payload,
    )
    log(f"Submitted job to hatchet.  Workflow run id: {workflowRef.workflow_run_id}")
    return {
        "workflow_run_id": workflowRef.workflow_run_id,
        "n_jobs": len(model_specs_df),
    }
