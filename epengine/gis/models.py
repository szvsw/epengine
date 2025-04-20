"""A module for DTO models etc for use in GIS jobs."""

from typing import Literal

from pydantic import BaseModel, Field

from epengine.models.leafs import WorkflowName


class GisJobAssumptions(BaseModel):
    """The assumptions for a GIS job to handle missing data."""

    wwr_ratio: float = Field(
        default=0.2,
        description="The window-to-wall ratio for the building when none is provided..",
    )
    num_floors: int = Field(
        default=2,
        description="The number of floors for the building when none is provided and no height is provided.",
    )
    f2f_height: float = Field(
        default=3.5,
        description="The floor-to-floor height; used when a building has no number of floors associated but does have a height, or neighter.",
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
