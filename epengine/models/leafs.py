"""Models for leaf workflows."""

from functools import cached_property
from typing import Literal

from pydantic import AnyUrl, Field

from epengine.experiments.tarkhan.models import TarkhanSpec
from epengine.models.base import LeafSpec
from epengine.models.shoebox import ShoeboxSimulationSpec
from epengine.models.shoebox_sbem import SBEMSimulationSpec
from epengine.models.train_sbem import TrainFoldSpec


class SimulationSpec(LeafSpec):
    """A spec for running an EnergyPlus simulation."""

    idf_uri: AnyUrl = Field(
        ..., description="The uri of the idf file to fetch and simulate"
    )
    epw_uri: AnyUrl = Field(
        ..., description="The uri of the epw file to fetch and simulate"
    )
    ddy_uri: AnyUrl | None = Field(
        default=None, description="The uri of the ddy file to fetch and simulate"
    )

    @cached_property
    def idf_path(self):
        """Fetch the idf file and return the local path.

        Returns:
            local_path (Path): The local path of the idf file
        """
        return self.fetch_uri(self.idf_uri)

    @cached_property
    def epw_path(self):
        """Fetch the epw file and return the local path.

        Returns:
            local_path (Path): The local path of the epw file
        """
        return self.fetch_uri(self.epw_uri)

    @cached_property
    def ddy_path(self):
        """Fetch the ddy file and return the local path.

        Returns:
            local_path (Path): The local path of the ddy file
        """
        if self.ddy_uri:
            return self.fetch_uri(self.ddy_uri)
        return None


class SimpleSpec(LeafSpec):
    """A test spec for working with a simple leaf workflow."""

    param_a: int = Field(..., description="A simple parameter A")


WorkflowName = Literal[
    "simulate_epw_idf",
    "simple",
    "simulate_ubem_shoebox",
    "simulate_sbem_shoebox",
    "train_regressor_with_cv_fold",
    "tarkhan",
]

AvailableWorkflowSpecs: dict[WorkflowName, type[LeafSpec]] = {
    "simulate_epw_idf": SimulationSpec,
    "simple": SimpleSpec,
    "simulate_ubem_shoebox": ShoeboxSimulationSpec,
    "simulate_sbem_shoebox": SBEMSimulationSpec,
    "train_regressor_with_cv_fold": TrainFoldSpec,
    "tarkhan": TarkhanSpec,
}
