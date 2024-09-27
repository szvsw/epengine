"""Models for lists and recursions of simulation specs."""

from typing import Generic, TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator

from epengine.models.base import BaseSpec, LeafSpec
from epengine.models.leaves import AvailableWorkflowSpecs, WorkflowName


class RecursionSpec(BaseModel):
    """A spec for recursive calls."""

    factor: int = Field(..., description="The factor to use in recursive calls", ge=1)
    offset: int | None = Field(
        default=None, description="The offset to use in recursive calls", ge=0
    )

    @model_validator(mode="before")
    @classmethod
    def validate_offset_less_than_factor(cls, values):
        """Validate that the offset is less than the factor."""
        if values["offset"] is None:
            return values
        if values["offset"] >= values["factor"]:
            raise ValueError(f"OFFSET:{values['offset']}>=FACTOR:{values['factor']}")
        return values


class RecursionMap(BaseModel):
    """A map of recursion specs to use in recursive calls.

    This allows a recursion node to understand where
    it is in the recursion tree and how to behave.
    """

    path: list[RecursionSpec] | None = Field(
        default=None, description="The path of recursion specs to use"
    )
    factor: int = Field(..., description="The factor to use in recursive calls", ge=1)
    max_depth: int = Field(
        default=10, description="The maximum depth of the recursion", ge=1, le=10
    )

    @field_validator("path", mode="before")
    @classmethod
    def validate_path_is_length_ge_1(cls, values):
        """Validate that the path is at least length 1."""
        if values is None:
            return values
        if len(values) < 1:
            raise ValueError("PATH:LENGTH:GE:1")
        return values


SpecListItem = TypeVar("SpecListItem", bound=LeafSpec)


class BranchesSpec(BaseSpec, Generic[SpecListItem]):
    """A spec for running multiple simulations.

    One key feature is that children simulations
    inherit the experiment_id of the parent simulations BaseSpec
    since they are both part of the same experiment.
    """

    specs: list[SpecListItem] = Field(
        ..., description="The list of simulation specs to run"
    )

    @model_validator(mode="before")
    @classmethod
    def set_children_exp_id_and_sort_index(cls, values: dict):
        """Set the experiment_id of each child spec to the experiment_id of the parent along with the sort_index."""
        if values.get("specs") is not None:
            for i, spec in enumerate(values["specs"]):
                if isinstance(spec, dict):
                    spec["experiment_id"] = values["experiment_id"]
                    if "sort_index" not in spec:
                        spec["sort_index"] = i
                elif isinstance(spec, LeafSpec):
                    spec.experiment_id = values["experiment_id"]
                    if "sort_index" not in spec.model_dump(
                        exclude_none=True, exclude_unset=True
                    ):
                        spec.sort_index = i
                else:
                    raise TypeError(f"SPEC:TYPE:{type(spec)}")

        return values


class WorkflowSelector(BaseModel, extra="ignore"):
    """A class for generating a BranchSpec from a workflow name."""

    workflow_name: WorkflowName = Field(
        ..., description="The name of the leaf workflow"
    )

    @property
    def Spec(self) -> type[LeafSpec]:
        """Return the spec class for the workflow.

        Returns:
            SpecClass (type[LeafSpec]): The spec class for the workflow
        """
        return AvailableWorkflowSpecs[self.workflow_name]

    @property
    def BranchesSpec(self) -> type[BranchesSpec[LeafSpec]]:
        """Return the branches spec class for the workflow.

        Returns:
            BranchesSpecClass (type[BranchesSpec[LeafSpec]]): The branches spec class for the workflow
        """
        return BranchesSpec[self.Spec]


if __name__ == "__main__":
    config = {
        "workflow_name": "simple",
        "other_junk": "junk",
    }

    leaf_workflow = WorkflowSelector.model_validate(config)
    print(leaf_workflow.BranchesSpec)
    print(
        leaf_workflow.BranchesSpec.model_validate({
            "specs": [{"param_a": 1}, {"param_a": 2}],
            "experiment_id": "exp_id",
        })
    )