"""Metadata for epws."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree

# Load the metadata
logger = logging.getLogger(__name__)

cached_epw_metadata_singleton: gpd.GeoDataFrame | None = None


def closest_epw(
    query_pts: gpd.GeoSeries,
    source_filter: str | None = None,
    crs: str | int = 3857,
    distance_threshold_meters: int | None = 500_000,
    metadata: gpd.GeoDataFrame | None = None,
    log_fn: Callable | None = None,
) -> gpd.GeoDataFrame:
    """Find the closest epw for each row in a dataframe."""
    # Get the closest epw for each row
    if metadata is None:
        global cached_epw_metadata_singleton
        if cached_epw_metadata_singleton is None:
            metadta_path = Path(__file__).parent / "epw_metadata.geojson"
            metadata = gpd.read_file(metadta_path)
            cached_epw_metadata_singleton = metadata
        else:
            metadata = cached_epw_metadata_singleton
    log = log_fn or logger.info
    log(f"Querying {len(query_pts)} points for the closest EPW.")
    filtered_metadata = (
        cast(gpd.GeoDataFrame, metadata.query(source_filter))
        if source_filter
        else metadata
    )

    # Get the closest epw for each row
    filtered_metadata_projected = cast(
        gpd.GeoDataFrame,
        filtered_metadata.to_crs(crs)
        if filtered_metadata.crs != crs
        else filtered_metadata,
    )
    points: gpd.GeoSeries = filtered_metadata_projected.geometry
    is_na = points.apply(
        lambda x: np.isinf(x.x) or np.isinf(x.y) or np.isnan(x.x) or np.isnan(x.y)
    )
    points = cast(gpd.GeoSeries, points[~is_na])
    points_tuples = points.apply(lambda x: (x.x, x.y)).tolist()
    filtered_metadata = filtered_metadata[~is_na]

    query_pts_projected = query_pts.to_crs(crs) if query_pts.crs != crs else query_pts
    query_points_proj = query_pts_projected.geometry
    query_points_tuples = query_points_proj.apply(lambda x: (x.x, x.y)).tolist()

    tree = cKDTree(points_tuples)
    distance, idx = tree.query(query_points_tuples)
    distance: np.ndarray

    # Return the closest epw
    selected_metadata: gpd.GeoDataFrame = filtered_metadata.iloc[idx]
    selected_metadata = cast(gpd.GeoDataFrame, selected_metadata.copy(deep=True))
    selected_metadata["distance"] = distance
    if distance_threshold_meters:
        mask = distance < distance_threshold_meters
        if not mask.all():
            log(f"Found {len(mask) - mask.sum()} points that are too far.")
            raise ValueError("EPW:TOO_FAR")
    selected_metadata = cast(
        gpd.GeoDataFrame, selected_metadata.set_index(query_pts.index)
    )
    log(f"Found {len(selected_metadata)} points.")
    return selected_metadata


if __name__ == "__main__":
    from shapely.geometry import Point

    massachusetts_lon_lat = (-71.3826, 42.4072)
    la_lon_lat = (-118.2437, 34.0522)
    test_pts = cast(
        gpd.GeoSeries,
        gpd.GeoSeries([Point(massachusetts_lon_lat), Point(la_lon_lat)]).set_crs(4326),
    )

    print(
        closest_epw(
            test_pts,
            source_filter="source in ['tmyx', 'tmy3']",
        )
    )
