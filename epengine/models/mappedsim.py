"""A model for a mapped simulation."""

import logging
from collections.abc import Callable
from functools import cached_property
from typing import cast

import numpy as np
import pandas as pd
import yaml
from archetypal.idfclass import IDF
from epinterface.geometry import (
    ShoeboxGeometry,
    compute_shading_mask,
    match_idf_to_building_and_neighbors,
)
from epinterface.sbem.builder import Model
from epinterface.sbem.components.composer import (
    construct_composer_model,
    construct_graph,
)
from epinterface.sbem.components.zones import ZoneComponent
from epinterface.sbem.prisma.client import PrismaSettings
from epinterface.weather import WeatherUrl
from pydantic import AnyUrl, Field

from epengine.models.base import LeafSpec

logger = logging.getLogger(__name__)


class SBEMSimulationSpec(LeafSpec):
    """A spec for running an EnergyPlus simulation."""

    db_uri: AnyUrl = Field(
        ...,
        description="The uri of the db file to providing the building component library.",
    )
    semantic_fields_uri: AnyUrl = Field(
        ...,
        description="The uri of the semantic fields file representing the expected semantic fields.",
    )
    component_map_uri: AnyUrl = Field(
        ...,
        description="The uri of the component map file representing how to compile the zone definition from components given the semantic field context.",
    )
    epwzip_uri: AnyUrl = Field(
        ..., description="The uri of the epw file to use during simulation."
    )
    # assumptions: GisJobAssumptions = Field(
    #     ..., description="The assumptions for the simulation"
    # )
    semantic_field_context: dict[str, float | str | int] = Field(
        ...,
        description="The semantic field values which will be used to compile the zone definition.",
    )
    neighbor_polys: list[str] = Field(
        ..., description="The polygons of the neighboring buildings."
    )
    neighbor_heights: list[float | int | None] = Field(
        ..., description="The height of the neighboring buildings  [m]."
    )
    neighbor_floors: list[float | int | None] = Field(
        ..., description="The number of floors of the neighboring buildings."
    )
    rotated_rectangle: str = Field(
        ..., description="The rotated rectangle fitted around the base of the building."
    )
    long_edge_angle: float = Field(
        ..., description="The long edge angle of the building (radians)."
    )
    long_edge: float = Field(
        ..., description="The length of the long edge of the building [m]."
    )
    short_edge: float = Field(
        ..., description="The length of the short edge of the building [m]."
    )
    aspect_ratio: float = Field(
        ..., description="The aspect ratio of the building footprint [unitless]."
    )
    rotated_rectangle_area_ratio: float = Field(
        ...,
        description="The ratio of the rotated rectangle footprint area to the building footprint area.",
    )
    wwr: float = Field(
        ..., description="The window-to-wall ratio of the building [unitless]."
    )
    height: float = Field(..., description="The height of the building [m].")
    num_floors: int = Field(..., description="The number of floors in the building.")
    f2f_height: float = Field(..., description="The floor to floor height [m].")
    # footprint_area: float = Field(
    #     ..., description="The footprint area of the building [m^2]."
    # )

    @property
    def feature_dict(self) -> dict[str, str | int | float]:
        """Return a dictionary of features which will be available to ML algos."""
        # Static features represent features which are constant for a given building and not subject to estimation, sampling, etc.
        features: dict[str, str | int | float] = {
            "feature.geometry.long_edge": self.long_edge,
            "feature.geometry.short_edge": self.short_edge,
            "feature.geometry.orientation": self.long_edge_angle,
            "feature.geometry.orientation.cos": np.cos(self.long_edge_angle),
            "feature.geometry.orientation.sin": np.sin(self.long_edge_angle),
            "feature.geometry.aspect_ratio": self.aspect_ratio,
            "feature.geometry.rotated_rectangle_area_ratio": self.rotated_rectangle_area_ratio,
            "feature.geometry.wwr": self.wwr,
            "feature.geometry.height": self.height,
            "feature.geometry.num_floors": self.num_floors,
            "feature.geometry.f2f_height": self.f2f_height,
            # TODO: add gfa? or just fp area? or pull from model.geometry?
        }

        # TODO: consider passing in
        # neighbors directly to Model.geometry, letting model perform neighbor
        # insertion directly rather than via a callback,
        # and then let shading mask become a computed property of the model.geometry.
        shading_mask = compute_shading_mask(
            self.rotated_rectangle,
            neighbors=self.neighbor_polys,
            neighbor_heights=self.neighbor_heights,
            azimuthal_angle=2 * np.pi / 48,
        )
        shading_mask_values = {
            f"feature.geometry.shading_mask_{i:02d}": val
            for i, val in enumerate(shading_mask.tolist())
        }
        features.update(shading_mask_values)

        # semantic features are kept separately as one building may have
        # multiple simulations with different semantic fields.
        features.update({
            f"feature.semantic.{feature_name}": feature_value
            for feature_name, feature_value in self.semantic_field_context.items()
        })

        features["feature.weather.file"] = self.epw_path.name

        # where to put ff2 height, wwr?

        # conditional features are derived from the static and semantic features,
        # and may be subject to things like conditional sampling, estimation etc.
        # e.g. rvalues, uvalues, schedule, etc.
        # additional things like basement/attic config?

        return features

    @cached_property
    def db_path(self):
        """Fetch the db file and return the local path.

        Returns:
            local_path (Path): The local path of the idf file
        """
        return self.fetch_uri(self.db_uri)

    @cached_property
    def semantic_fields_path(self):
        """Fetch the semantic fields file and return the local path.

        Returns:
            local_path (Path): The local path of the semantic fields file
        """
        return self.fetch_uri(self.semantic_fields_uri)

    @cached_property
    def epw_path(self):
        """Fetch the epw file and return the local path.

        Returns:
            local_path (Path): The local path of the epw file
        """
        return self.fetch_uri(self.epwzip_uri)

    @property
    def component_map(self):
        """Fetch the component map file and return the local path.

        Returns:
            local_path (Path): The local path of the component map file
        """
        return self.fetch_uri(self.component_map_uri)

    def construct_zone_def(self) -> ZoneComponent:
        """Construct the zone definition for the simulation.

        Returns:
            zone_def (ZoneComponent): The zone definition for the simulation
        """
        g = construct_graph(ZoneComponent)
        SelectorModel = construct_composer_model(
            g,
            ZoneComponent,
            use_children=False,
        )

        with open(self.component_map) as f:
            component_map_yaml = yaml.safe_load(f)
        selector = SelectorModel.model_validate(component_map_yaml)

        # TODO: make sure we are okay with accwssing the same db
        # across workers executing the same experiment.
        settings = PrismaSettings.New(
            database_path=self.db_path, if_exists="ignore", auto_register=False
        )
        db = settings.db

        context = self.semantic_field_context
        with db:
            return cast(ZoneComponent, selector.get_component(context=context, db=db))

    def run(self, log_fn: Callable | None = None):
        """Run the simulation.

        Args:
            log_fn (Callable | None): The function to use for logging.
        """
        _log = log_fn or logger.info
        zone_def = self.construct_zone_def()

        # TODO: Geometry loading, weather loading, geometry post process, etc
        model = Model(
            Weather=WeatherUrl(self.epwzip_uri),  # pyright: ignore [reportCallIssue]
            Zone=zone_def,
            # TODO: compute these somehow?
            basement_insulation_surface=None,
            conditioned_basement=False,
            basement_use_fraction=None,
            attic_insulation_surface=None,
            conditioned_attic=False,
            attic_use_fraction=None,
            geometry=ShoeboxGeometry(
                x=0,
                y=0,
                w=10,
                d=10,
                h=self.f2f_height,
                wwr=self.wwr,
                num_stories=self.num_floors,
                basement=False,
                zoning="by_storey",  # TODO: determine zoning based off of fp area, can be done in geometry itself?
                roof_height=None,
            ),
        )

        # TODO: implement post-build callback.
        # and maybe move it into epinterface.geometry
        # using neighboring context as argument to
        # model.geometry.
        def post_geometry_callback(idf: IDF) -> IDF:
            idf = match_idf_to_building_and_neighbors(
                idf,
                building=self.rotated_rectangle,
                neighbor_polys=self.neighbor_polys,  # pyright: ignore [reportArgumentType]
                neighbor_floors=[
                    (floor / self.f2f_height) if floor is not None else None
                    for floor in self.neighbor_floors
                ],
                neighbor_f2f_height=self.f2f_height,
                target_short_length=self.short_edge,
                target_long_length=self.long_edge,
                rotation_angle=self.long_edge_angle,
            )
            return idf

        idf, results, err_text = model.run(
            move_energy=False, post_geometry_callback=post_geometry_callback
        )
        # creating results dataframe index

        dumped_self = self.model_dump(
            mode="json",
            exclude={
                "feature_dict",
                "neighbor_polys",
                "neighbor_floors",
                "neighbor_heights",
                "semantic_field_context",
            },
        )
        dumped_self.update(self.feature_dict)
        dumped_self["feature.geometry.core_zone_split"] = model.geometry.zoning

        index = pd.MultiIndex.from_tuples(
            [tuple(dumped_self.values())],
            names=list(dumped_self.keys()),
        )
        results = results.to_frame().T.set_index(index)
        return idf, results, err_text
