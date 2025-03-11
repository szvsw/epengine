"""A model for a mapped simulation."""

import logging
from collections.abc import Callable
from functools import cached_property
from typing import Any, cast

import yaml
from epinterface.sbem.components.composer import (
    construct_composer_model,
    construct_graph,
)
from epinterface.sbem.components.zones import ZoneComponent
from epinterface.sbem.prisma.client import PrismaSettings
from prisma import Prisma
from pydantic import AnyUrl, Field

from epengine.models.base import LeafSpec

logger = logging.getLogger(__name__)


class SBEMSimulationSpec(LeafSpec):
    """A spec for running an EnergyPlus simulation."""

    db_uri: AnyUrl = Field(
        ..., description="The uri of the idf file to fetch and simulate"
    )
    semantic_fields_uri: AnyUrl = Field(
        ..., description="The uri of the epw file to fetch and simulate"
    )
    component_map_uri: AnyUrl = Field(
        ..., description="The uri of the component map file to fetch and simulate"
    )
    ddy_uri: AnyUrl = Field(
        ..., description="The uri of the ddy file to fetch and simulate"
    )
    epw_uri: AnyUrl = Field(
        ..., description="The uri of the epw file to fetch and simulate"
    )
    # assumptions: GisJobAssumptions = Field(
    #     ..., description="The assumptions for the simulation"
    # )

    semantic_field_context: dict[str, Any] = Field(
        ..., description="The context for the semantic fields"
    )

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
    def ddy_path(self):
        """Fetch the ddy file and return the local path.

        Returns:
            local_path (Path): The local path of the ddy file
        """
        return self.fetch_uri(self.ddy_uri)

    @cached_property
    def epw_path(self):
        """Fetch the epw file and return the local path.

        Returns:
            local_path (Path): The local path of the epw file
        """
        return self.fetch_uri(self.epw_uri)

    @property
    def component_map(self):
        """Fetch the component map file and return the local path.

        Returns:
            local_path (Path): The local path of the component map file
        """
        return self.fetch_uri(self.component_map_uri)

    def construct_zone_def(self, db: Prisma) -> ZoneComponent:
        """Construct the zone definition for the simulation.

        Args:
            db (Prisma): The database to use for the simulation

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

        context = self.semantic_field_context
        return cast(ZoneComponent, selector.get_component(context=context, db=db))

    def run(self, log_fn: Callable | None = None):
        """Run the simulation.

        Args:
            log_fn (Callable | None): The function to use for logging.
        """
        log = log_fn or logger.info
        settings = PrismaSettings.New(
            database_path=self.db_path, if_exists="ignore", auto_register=False
        )
        db = settings.db
        with db:
            zone_def = self.construct_zone_def(db)
        log(f"Zone definition: {zone_def}")
        return 0
