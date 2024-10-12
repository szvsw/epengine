"""A module containing the scatter-gather workflow and related classes."""

import asyncio
import tempfile
from collections.abc import Coroutine
from typing import Any, Generic

import boto3
import pandas as pd
from hatchet_sdk import Context
from hatchet_sdk.workflow_run import WorkflowRunRef
from pydantic import AnyUrl, BaseModel, Field

from epengine.hatchet import hatchet
from epengine.models.branches import (
    BranchesSpec,
    RecursionMap,
    RecursionSpec,
    SpecListItem,
    WorkflowSelector,
)
from epengine.models.mixins import WithBucket, WithHContext, WithOptionalBucket
from epengine.models.outputs import URIResponse
from epengine.utils.results import (
    collate_subdictionaries,
    combine_recurse_results,
    create_errored_and_missing_df,
    save_and_upload_results,
    separate_errors_and_safe_sim_results,
    serialize_df_dict,
)

s3 = boto3.client("s3")


class ScatterGatherSpec(
    WithHContext, BranchesSpec[SpecListItem], Generic[SpecListItem]
):
    """A class representing the specs for a scatter-gather workflow."""

    pass


class ScatterGatherSpecWithOptionalBucket(
    WithOptionalBucket, ScatterGatherSpec[SpecListItem], Generic[SpecListItem]
):
    """A class representing the specs for a scatter-gather workflow with an optional bucket (non recursive)."""

    pass


class RecursionId(BaseModel):
    """A class representing the recursion ID for a scatter-gather recursive workflow."""

    index: int
    level: int


class ScatterGatherRecursiveSpec(
    WithBucket, ScatterGatherSpec[SpecListItem], Generic[SpecListItem]
):
    """A class representing the specs for a scatter-gather recursive workflow."""

    recursion_map: RecursionMap = Field(
        ...,
        description="The recursion map to use in the scatter-gather workflow",
    )

    @property
    def output_key(self):
        """Returns the output key for the results of the scatter-gather workflow.

        Note that this is based on the recursion map, and will place the results in
        a directory based on the recursion level.

        Returns:
            str: The output key for the results of the scatter-gather workflow.
        """
        output_key_base = f"hatchet/{self.experiment_id}/results"
        output_results_dir = (
            f"recurse/{len(self.recursion_map.path)}"
            if self.recursion_map.path
            else "root"
        )
        output_key_prefix = f"{output_key_base}/{output_results_dir}"
        workflow_run_id = self.hcontext.workflow_run_id()
        output_key = f"{output_key_prefix}/{workflow_run_id}.h5"
        return output_key

    @property
    def selected_specs(self):
        """Returns the selected specs based on the recursion map.

        Returns:
            list[BranchesSpec[LeafSpec]]: The selected specs based on the recursion map.
        """
        spec_list = self.specs
        path = self.recursion_map.path

        if self.recursion_map.specs_already_selected:
            return spec_list

        # if path is present, we need to trim down
        if path:
            # we are in a recursive call and so we
            # need to trim down the spec list
            for recur in path:
                spec_list = spec_list[recur.offset :: recur.factor]
        return spec_list

    async def recurse(self, workflow_input: dict) -> URIResponse:
        """Recursively executes the scatter-gather workflow.

        Note that this spawns new workflows from the current context.

        Args:
            workflow_input (dict): The input data for the workflow.

        Returns:
            URIResponse: The response containing the URI of the saved results.
        """
        tasks: list[WorkflowRunRef] = []
        task_specs: list[RecursionId] = []
        task_ids: list[str] = []
        select_specs = True
        for i in range(self.recursion_map.factor):
            payload = workflow_input.copy()
            new_path = self.recursion_map.path.copy() if self.recursion_map.path else []
            new_path.append(RecursionSpec(factor=self.recursion_map.factor, offset=i))
            if select_specs:
                specs_to_use = self.selected_specs[i :: self.recursion_map.factor]
                specs_as_df = pd.DataFrame([
                    spec.model_dump(mode="json") for spec in specs_to_use
                ])
                with tempfile.TemporaryDirectory() as tmpdir:
                    f = f"{tmpdir}/specs.parquet"
                    specs_as_df.to_parquet(f)
                    # TODO: fix naming here.
                    run_id = self.hcontext.workflow_run_id()
                    key = f"hatchet/{self.experiment_id}/specs/{run_id}/{run_id}_specs_{i:06d}.pq"
                    uri = f"s3://{self.bucket}/{key}"
                    s3.upload_file(f, self.bucket, key)
                del specs_as_df
                del specs_to_use
                payload.update({
                    "specs": uri,
                })
            recurse = RecursionMap(
                path=new_path,
                factor=self.recursion_map.factor,
                max_depth=self.recursion_map.max_depth,
                specs_already_selected=select_specs,
            )
            payload.update({
                "recursion_map": recurse.model_dump(mode="json"),
            })
            recursion_id = RecursionId(index=i, level=len(new_path))
            # new_specs = ScatterGatherRecursiveSpec(**workflow_input)
            task_specs.append(recursion_id)
            task = await self.hcontext.aio.spawn_workflow(
                "scatter_gather_recursive",
                # new_specs.model_dump(mode="json"),
                payload,
                options={
                    "additional_metadata": {
                        **recursion_id.model_dump(),
                        "factor": self.recursion_map.factor,  # pyright: ignore [reportArgumentType]
                    }
                },
            )
            task_id = task.workflow_run_id
            task_ids.append(task_id)
            tasks.append(task)

        recurse_task_promises = [task.result() for task in tasks]
        self.log(f"Waiting for {len(recurse_task_promises)} children to complete...")
        results = await asyncio.gather(*recurse_task_promises, return_exceptions=True)
        self.log("Children have completed!")

        safe_results, errored_results = separate_errors_and_safe_sim_results(
            task_ids,
            task_specs,
            results,
        )

        # Log the children recurses which errored
        for workflow_id, *_ in errored_results:
            self.log(f"WORKFLOW_ID: {workflow_id}")
            self.log("Error in recursive scatter-gather!")

        # TODO: some type safety here on the result objects so they conform to a spec would be
        # nice
        step_results: list[dict[str, Any]] = [
            step
            for workflow_id, task_spec, result in safe_results
            for (step_name, step) in result.items()
            if step_name == "spawn_children"
        ]
        self.log("Combining results...")
        collected_dfs = combine_recurse_results(step_results)
        self.log("Results combined!")

        # serialize the dataframes and save to s3
        self.log("Saving results...")
        uri = save_and_upload_results(
            collected_dfs,
            s3=s3,
            bucket=self.bucket,
            output_key=self.output_key,
        )
        del collected_dfs
        self.log("Results saved!")

        return URIResponse(uri=AnyUrl(uri))


@hatchet.workflow(
    name="scatter_gather",
    version="0.2",
    timeout="1200m",
    on_events=["simulations:execute"],
)
class ScatterGatherWorkflow:
    """A scatter-gather workflow that executes simulations in a non-recursive manner."""

    @hatchet.step(timeout="1200m")
    async def spawn_children(self, context: Context):
        """Spawns the children workflows for the scatter-gather workflow.

        Note that this will execute the simulations in a non-recursive manner.

        Args:
            context (Context): The context of the workflow.

        Returns:
            dict: The results of the simulations.  This may be a dictionary of DataFrames or a URIResponse.

        """
        workflow_input = context.workflow_input()
        workflow_selection = WorkflowSelector.model_validate(workflow_input)
        specs_without_context = workflow_selection.BranchesSpec.from_payload(
            workflow_input
        )
        specs = ScatterGatherSpecWithOptionalBucket[workflow_selection.Spec](
            **specs_without_context.model_dump(),
            hcontext=context,
        )

        return await execute_simulations(workflow_selection, specs)


@hatchet.workflow(
    name="scatter_gather_recursive",
    version="0.2",
    timeout="1200m",
    on_events=["simulations:recursive"],
    schedule_timeout="240m",
)
class ScatterGatherRecursiveWorkflow:
    """A scatter-gather workflow that executes simulations in a recursive manner."""

    @hatchet.step(timeout="1200m")
    async def spawn_children(self, context: Context):
        """Spawns the children workflows for the recursive scatter-gather workflow.

        Args:
            context (Context): The context of the workflow.

        Returns:
            URIResponse: The response containing the URI of the saved results.
        """
        workflow_input = context.workflow_input()

        workflow_selection = WorkflowSelector.model_validate(workflow_input)
        specs_without_context = workflow_selection.BranchesSpec.from_payload(
            workflow_input
        )
        manager = ScatterGatherRecursiveSpec[workflow_selection.Spec](
            **specs_without_context.model_dump(exclude={"recursion_map"}),
            hcontext=context,
            recursion_map=workflow_input["recursion_map"],
        )
        if len(manager.recursion_map.path or []) > 10:
            raise ValueError("RecusionDepthExceeded")

        selected_specs = manager.selected_specs

        # Random error for testing purposes
        # import numpy as np
        # if manager.recursion_map.path and np.random.random() < 0.5:
        #     raise ValueError("RandomError")

        # Check to see if we have hit the base case
        too_few_specs = len(selected_specs) <= manager.recursion_map.factor
        past_max_depth = (
            (len(manager.recursion_map.path) >= manager.recursion_map.max_depth)
            if manager.recursion_map.path
            else False
        )
        if too_few_specs or past_max_depth:
            # we are at a base case which should be run
            data = {**manager.model_dump(), "specs": selected_specs}
            new_specs = ScatterGatherSpecWithOptionalBucket[workflow_selection.Spec](
                **data,
                hcontext=context,
            )
            result = await execute_simulations(workflow_selection, new_specs)
            return result
        else:
            result = await manager.recurse(workflow_input)
            return result.model_dump(mode="json")


async def execute_simulations(
    workflow_selection: WorkflowSelector,
    specs: ScatterGatherSpecWithOptionalBucket[SpecListItem],
):
    """Executes the simulations in the given specs.

    Note that this requires the use of the Hatchet context. All simulations will be spawned (non-recursive).

    Args:
        workflow_selection (WorkflowSelector): The workflow selection.
        specs (ScatterGatherSpecWithOptionalBucket): The specs to execute.

    Returns:
        dict: The results of the simulations.  This may be a dictionary of DataFrames or a URIResponse.
    """
    # TODO: this should get its own in config

    promises: list[Coroutine[Any, Any, dict[str, Any]]] = []
    ids: list[str] = []

    for i, spec in enumerate(specs.specs):
        if i % 1000 == 0:
            specs.hcontext.log(
                f"Queuing {i}th child workflow out of {len(specs.specs)}..."
            )
        task = await specs.hcontext.aio.spawn_workflow(
            workflow_selection.workflow_name,
            spec.model_dump(mode="json"),
            options={
                "additional_metadata": {
                    "index": i,  # pyright: ignore [reportArgumentType]
                },
            },
        )
        # TODO: check error handling
        promise = task.result()
        promises.append(promise)
        ids.append(task.workflow_run_id)

    specs.log(f"Waiting for {len(promises)} children to complete...")
    results = await asyncio.gather(*promises, return_exceptions=True)
    specs.log("Children have completed!")

    # get the results
    safe_results, errored_workflows = separate_errors_and_safe_sim_results(
        ids,
        specs.specs,
        results,
    )
    sim_results = [
        result["simulate"] for _, _, result in safe_results if "simulate" in result
    ]

    # combine the successful results
    specs.log("Combining results...")
    collated_dfs = collate_subdictionaries(sim_results)

    # create the missing results and add to main results
    missing_results = [
        (workflow_id, spec)
        for workflow_id, spec, result in safe_results
        if "simulate" not in result
    ]
    errored_df = create_errored_and_missing_df(errored_workflows, missing_results)
    if len(errored_df) > 0:
        collated_dfs["runtime_errors"] = errored_df

    specs.log("Results combined!")

    # Finish up
    if specs.bucket:
        # save as hdfs
        workflow_run_id = specs.hcontext.workflow_run_id()
        # TODO: hatchet prefix should come from task!
        output_key = f"hatchet/{specs.experiment_id}/results/{workflow_run_id}.h5"
        specs.log("Saving results...")
        uri = save_and_upload_results(
            collated_dfs,
            s3=s3,
            bucket=specs.bucket,
            output_key=output_key,
        )
        del collated_dfs
        specs.log("Results saved!")
        return URIResponse(uri=AnyUrl(uri)).model_dump(mode="json")

    else:
        dfs = serialize_df_dict(collated_dfs)
        return dfs
