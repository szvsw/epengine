"""A module containing the scatter-gather workflow and related classes."""

import asyncio
import tempfile
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
    ZipDataContent,
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

    async def recurse(
        self, workflow_input: dict
    ) -> tuple[list[WorkflowRunRef], list[RecursionId], list[str]]:
        """Recursively executes the scatter-gather workflow.

        Note that this spawns new workflows from the current context.

        Args:
            workflow_input (dict): The input data for the workflow.

        Returns:
            tasks (list[WorkflowRunRef]): The tasks that were spawned.
            task_specs (list[RecursionId]): The specs for the tasks.
            task_ids (list[str]): The IDs of the tasks that were spawned.
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
        return tasks, task_specs, task_ids

    async def collect_tasks(
        self, task_ids: list[str], task_specs: list[ZipDataContent], leafs: bool
    ) -> URIResponse:
        """Collects the results of the scatter-gather recursive workflow.

        Args:
            task_ids (list[str]): The IDs of the tasks to collect.
            task_specs (list[ZipDataContent]): The specs for the tasks.
            leafs (bool): Whether or not this is a terminal leaf collection.

        Returns:
            URIResponse: The response containing the URI of the saved results.
        """
        tasks = [self.hcontext.admin_client.get_workflow_run(id_) for id_ in task_ids]
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
            if leafs
            or (step_name == "collect_children")  # TODO: better selection of step name
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

    async def collect(self):
        """Collects the results of the scatter-gather recursive workflow.

        Returns:
            result (URIResponse): The response containing the URI of the saved results.
        """
        step_output = SpawnResult.model_validate(
            self.hcontext.step_output("spawn_children")
        )
        if step_output.recurse_specs is not None:
            result = await self.collect_tasks(
                step_output.children_ids, step_output.recurse_specs, leafs=False
            )
        else:
            specs_to_collect = self.selected_specs
            result = await self.collect_tasks(
                step_output.children_ids, specs_to_collect, leafs=True
            )
            # TODO: should we be using the regular collect specs here?

        return result

    async def spawn(self, workflow_selection: WorkflowSelector):
        """Spawns the children workflows for the scatter-gather recursive workflow.

        Args:
            workflow_selection (WorkflowSelector): The workflow selection.

        Returns:
            children_ids (SpawnResult): The IDs of the children workflows as dict.
        """
        workflow_input = self.hcontext.workflow_input()
        if len(self.recursion_map.path or []) > 10:
            raise ValueError("RecusionDepthExceeded")

        selected_specs = self.selected_specs

        # Random error for testing purposes
        # import numpy as np
        # if manager.recursion_map.path and np.random.random() < 0.5:
        #     raise ValueError("RandomError")

        # Check to see if we have hit the base case
        too_few_specs = len(selected_specs) <= self.recursion_map.factor
        past_max_depth = (
            (len(self.recursion_map.path) >= self.recursion_map.max_depth)
            if self.recursion_map.path
            else False
        )
        if too_few_specs or past_max_depth:
            # we are at a base case which should be run
            data = {**self.model_dump(), "specs": selected_specs}
            new_specs = ScatterGatherSpecWithOptionalBucket[workflow_selection.Spec](
                **data,
                hcontext=self.hcontext,
            )
            # result = await execute_simulations(workflow_selection, new_specs)
            # return result
            tasks, ids = await spawn_simulations(workflow_selection, new_specs)
            result = SpawnResult(children_ids=ids)
            return result
        else:
            tasks, recurse_specs, ids = await self.recurse(workflow_input)
            result = SpawnResult(children_ids=ids, recurse_specs=recurse_specs)
            return result


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

        _tasks, ids = await spawn_simulations(workflow_selection, specs)
        # TODO: serialize to a file and upload to s3 and return a uri
        result = SpawnResult(children_ids=ids)
        return result.model_dump(mode="json")

    @hatchet.step(timeout="120m", parents=["spawn_children"])
    async def collect_children(self, context: Context):
        """Collects the results of the scatter-gather workflow.

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
        step_output = SpawnResult.model_validate(context.step_output("spawn_children"))
        tasks = [
            context.admin_client.get_workflow_run(id_)
            for id_ in step_output.children_ids
        ]
        results = await collect_simulations(tasks, step_output.children_ids, specs)
        return results


class SpawnResult(BaseModel):
    """A class representing the result of a spawn operation."""

    children_ids: list[str]
    recurse_specs: list[RecursionId] | None = None


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
            children_ids (SpawnResult): The IDs of the children workflows as dict.
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
        result = await manager.spawn(workflow_selection)
        return result.model_dump(mode="json")

    @hatchet.step(timeout="120m", parents=["spawn_children"])
    async def collect_children(self, context: Context):
        """Collects the results of the scatter-gather recursive workflow.

        Args:
            context (Context): The context of the workflow.

        Returns:
            result (URIResponse): The response containing the URI of the saved results as dict
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

        result = await manager.collect()
        return result.model_dump(mode="json")


async def spawn_simulations(
    workflow_selection: WorkflowSelector,
    specs: ScatterGatherSpecWithOptionalBucket[SpecListItem],
):
    """Spawns the simulations in the given specs.

    Note that this requires the use of the Hatchet context. All simulations will be spawned (non-recursive).

    Args:
        workflow_selection (WorkflowSelector): The workflow selection.
        specs (ScatterGatherSpecWithOptionalBucket): The specs to spawn.

    Returns:
        tasks (list[WorkflowRunRef]): The tasks that were spawned.
        ids (list[str]): The IDs of the tasks that were spawned.
    """
    tasks: list[WorkflowRunRef] = []
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
        ids.append(task.workflow_run_id)
        tasks.append(task)
    return tasks, ids


async def collect_simulations(
    tasks: list[WorkflowRunRef],
    ids: list[str],
    specs: ScatterGatherSpecWithOptionalBucket[SpecListItem],
):
    """Collects the simulations in the given specs.

    Note that this requires the use of the Hatchet context. All simulations will be collected (non-recursive).

    Args:
        tasks (list[WorkflowRunRef]): The tasks to collect.
        ids (list[str]): The IDs of the tasks to collect.
        specs (ScatterGatherSpecWithOptionalBucket): The specs to collect.

    Returns:
        dict: The results of the simulations.  This may be a dictionary of DataFrames or a URIResponse.
    """
    promises = [task.result() for task in tasks]
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
    tasks, ids = await spawn_simulations(workflow_selection, specs)
    return await collect_simulations(tasks, ids, specs)
