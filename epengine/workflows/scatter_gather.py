"""A module containing the scatter-gather workflow and related classes."""

import asyncio
import tempfile
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, TypeVar, cast

import boto3
import pandas as pd
from hatchet_sdk import Context
from hatchet_sdk.workflow_run import WorkflowRunRef
from pydantic import AnyUrl, BaseModel, Field

from epengine.hatchet import hatchet
from epengine.models.configs import (
    RecursionMap,
    RecursionSpec,
    SimulationSpec,
    SimulationsSpec,
    URIResponse,
    WithBucket,
    WithHContext,
    WithOptionalBucket,
    fetch_uri,
)
from epengine.utils.results import collate_subdictionaries, serialize_df_dict

s3 = boto3.client("s3")


class ScatterGatherSpec(WithHContext, SimulationsSpec):
    """A class representing the specs for a scatter-gather workflow."""

    pass


class ScatterGatherSpecWithOptionalBucket(WithOptionalBucket, ScatterGatherSpec):
    """A class representing the specs for a scatter-gather workflow with an optional bucket (non recursive)."""

    pass


class RecursionId(BaseModel):
    """A class representing the recursion ID for a scatter-gather recursive workflow."""

    index: int
    level: int


class ScatterGatherRecursiveSpec(WithBucket, ScatterGatherSpec):
    """A class representing the specs for a scatter-gather recursive workflow."""

    recursion_map: RecursionMap = Field(..., description="The recursion map to use in the scatter-gather workflow")

    @property
    def output_key(self):
        """Returns the output key for the results of the scatter-gather workflow.

        Note that this is based on the recursion map, and will place the results in
        a directory based on the recursion level.

        Returns:
            str: The output key for the results of the scatter-gather workflow.
        """
        output_key_base = f"hatchet/{self.experiment_id}/results"
        output_results_dir = f"recurse/{len(self.recursion_map.path)}" if self.recursion_map.path else "root"
        output_key_prefix = f"{output_key_base}/{output_results_dir}"
        workflow_run_id = self.hcontext.workflow_run_id()
        output_key = f"{output_key_prefix}/{workflow_run_id}.h5"
        return output_key

    @property
    def selected_specs(self):
        """Returns the selected specs based on the recursion map.

        Returns:
            list[SimulationSpec]: The selected specs based on the recursion map.
        """
        spec_list = self.specs
        path = self.recursion_map.path
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
        for i in range(self.recursion_map.factor):
            new_path = self.recursion_map.path.copy() if self.recursion_map.path else []
            new_path.append(RecursionSpec(factor=self.recursion_map.factor, offset=i))
            recurse = RecursionMap(path=new_path, factor=self.recursion_map.factor)
            payload = workflow_input.copy()
            payload.update({"recursion_map": recurse.model_dump(mode="json")})
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

        safe_results, errored_results = separate_errors_and_safe_sim_results(task_ids, task_specs, results)

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
        specs_without_context = SimulationsSpec.from_payload(workflow_input)
        specs = ScatterGatherSpecWithOptionalBucket(**specs_without_context.model_dump(), hcontext=context)

        return await execute_simulations(specs)


@hatchet.workflow(
    name="scatter_gather_recursive",
    version="0.2",
    timeout="1200m",
    on_events=["simulations:recursive"],
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

        specs_without_context = SimulationsSpec.from_payload(workflow_input)
        manager = ScatterGatherRecursiveSpec(
            **specs_without_context.model_dump(),
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
        if len(selected_specs) <= manager.recursion_map.factor:
            # we are at a base case which should be run
            data = {**manager.model_dump(), "specs": selected_specs}
            new_specs = ScatterGatherSpecWithOptionalBucket(
                **data,
                hcontext=context,
            )
            result = await execute_simulations(new_specs)
            return result
        else:
            result = await manager.recurse(workflow_input)
            return result.model_dump(mode="json")


async def execute_simulations(specs: ScatterGatherSpecWithOptionalBucket):
    """Executes the simulations in the given specs.

    Note that this requires the use of the Hatchet context. All simulations will be spawned (non-recursive).

    Args:
        specs (ScatterGatherSpecWithOptionalBucket): The specs to execute.

    Returns:
        dict: The results of the simulations.  This may be a dictionary of DataFrames or a URIResponse.
    """
    # TODO: this should get its own in config

    promises: list[Coroutine[Any, Any, dict[str, Any]]] = []
    ids: list[str] = []

    for i, spec in enumerate(specs.specs):
        if i % 1000 == 0:
            specs.hcontext.log(f"Queing {i}th child workflow...")
        task = await specs.hcontext.aio.spawn_workflow(
            "simulate_epw_idf",
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
    safe_results, errored_workflows = separate_errors_and_safe_sim_results(ids, specs.specs, results)
    sim_results = [result["simulate"] for _, _, result in safe_results if "simulate" in result]

    # combine the successful results
    specs.log("Combining results...")
    collated_dfs = collate_subdictionaries(sim_results)

    # create the missing results and add to main results
    missing_results = [(workflow_id, spec) for workflow_id, spec, result in safe_results if "simulate" not in result]
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
            bucket=specs.bucket,
            output_key=output_key,
        )
        del collated_dfs
        specs.log("Results saved!")
        return URIResponse(uri=AnyUrl(uri)).model_dump(mode="json")

    else:
        dfs = serialize_df_dict(collated_dfs)
        return dfs


ZipDataContent = TypeVar("ZipDataContent", SimulationSpec, RecursionId)
ResultDataContent = TypeVar("ResultDataContent")


def separate_errors_and_safe_sim_results(
    ids: list[str],
    zip_data: list[ZipDataContent],
    results: list[ResultDataContent | BaseException],
):
    """Separates errored workflows from safe simulation results.

    Args:
        ids (list[str]): The list of workflow IDs.
        zip_data (list[ZipDataContent]): The list of data to return with rows.
        results (list[ResultDataContent | BaseException]): The list of results to separate.

    Returns:
        tuple[list[tuple[str, ZipDataContent, ResultDataContent]], list[tuple[str, ZipDataContent, BaseException]]]: A tuple containing the safe results and errored workflows.
    """
    errored_workflows: list[tuple[str, ZipDataContent, BaseException]] = []
    safe_results: list[tuple[str, ZipDataContent, ResultDataContent]] = []
    for workflow_id, spec, result in zip(ids, zip_data, results, strict=True):
        if isinstance(result, BaseException):
            errored_workflows.append((workflow_id, spec, result))
        else:
            safe_results.append((workflow_id, spec, result))
    return safe_results, errored_workflows


def create_errored_and_missing_df(
    errored_workflows: list[tuple[str, ZipDataContent, BaseException]],
    missing_results: list[tuple[str, ZipDataContent]],
):
    """Creates a DataFrame of errored and missing results.

    Args:
        errored_workflows (list[tuple[str, ZipDataContent, BaseException]]): The list of errored workflows.
        missing_results (list[tuple[str, ZipDataContent]]): The list of missing results.

    Returns:
        pd.DataFrame: The DataFrame of errored and missing results.
    """
    error_dfs: list[pd.DataFrame] = []
    for child_workflow_run_id, spec, result in errored_workflows:
        index_data = spec.model_dump(mode="json")
        index_data.update({"workflow_run_id": child_workflow_run_id})
        index = pd.MultiIndex.from_tuples([tuple(index_data.values())], names=list(index_data.keys()))
        row_data = {"missing": [False], "errored": [str(result)]}
        df = pd.DataFrame(row_data, index=index)
        error_dfs.append(df)

    for workflow_id, spec in missing_results:
        index_data = spec.model_dump(mode="json")
        index_data.update({"workflow_run_id": workflow_id})
        index = pd.MultiIndex.from_tuples([tuple(index_data.values())], names=list(index_data.keys()))
        row_data = {"missing": [True], "errored": [None]}
        df = pd.DataFrame(row_data, index=index)
        error_dfs.append(df)
    error_df = pd.concat(error_dfs) if error_dfs else pd.DataFrame({"missing": [], "errored": []})
    return error_df


def save_and_upload_results(
    collected_dfs: dict[str, pd.DataFrame], bucket: str, output_key: str, save_errors: bool = False
):
    """Saves and uploads the collected dataframes to S3.

    Args:
        collected_dfs (dict[str, pd.DataFrame]): A dictionary containing the collected dataframes.
        bucket (str): The S3 bucket to upload the results to.
        output_key (str): The key to use for the uploaded results.
        save_errors (bool): Whether to save errors in the results.

    Returns:
        uri (str): The URI of the uploaded results.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        local_path = f"{tempdir}/results.h5"
        for key, df in collected_dfs.items():
            if "error" in key.lower() and not save_errors:
                continue
            df.to_hdf(local_path, key=key, mode="a")
        s3.upload_file(Bucket=bucket, Key=output_key, Filename=local_path)
        uri = f"s3://{bucket}/{output_key}"
        return uri


# TODO: consider using a list based method which
# concats all at once rather than repeated concats
def update_collected_with_df(collected_dfs: dict[str, pd.DataFrame], key: str, df: pd.DataFrame):
    """Updates the collected dataframes with a new DataFrame.

    Note that this function mutates the collected_dfs dictionary and does
    not return a value.

    Args:
        collected_dfs (dict[str, pd.DataFrame]): A dictionary containing the collected dataframes.
        key (str): The key to use for the new DataFrame.
        df (pd.DataFrame): The DataFrame to add to the collected dataframes.

    Returns:
        None (mutates the collected_dfs dictionary).
    """
    if key in collected_dfs:
        df = pd.concat([collected_dfs[key], df], axis=0)
    collected_dfs[key] = df


def handle_explicit_result(collected_dfs: dict[str, pd.DataFrame], result: dict):
    """Updates the collected dataframes with an explicit result.

    Note that this function mutates the collected_dfs dictionary and does
    not return a value.

    Args:
        collected_dfs (dict[str, pd.DataFrame]): A dictionary containing the collected dataframes.
        result (dict): The explicit result to handle.

    Returns:
        None (mutates the collected_dfs dictionary).
    """
    for key, df_dict in result.items():
        df = pd.DataFrame.from_dict(df_dict, orient="tight")
        update_collected_with_df(collected_dfs, key, df)
        del df, df_dict


def handle_referenced_result(collected_dfs: dict[str, pd.DataFrame], uri: str):
    """Fetches a result from a given URI and updates the collected dataframes.

    Note that this function mutates the collected_dfs dictionary and does
    not return a value.

    Args:
        collected_dfs (dict[str, pd.DataFrame]): A dictionary containing the collected dataframes.
        uri (str): The URI of the result to fetch.

    Returns:
        None (mutates the collected_dfs dictionary).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        res_path = fetch_uri(uri=uri, use_cache=False, local_path=Path(tmpdir) / "result.h5")
        # list the keys in the h5 file
        with pd.HDFStore(res_path, mode="r") as store:
            keys = store.keys()
            for key in keys:
                # get the dataframes
                df = cast(pd.DataFrame, store[key])
                update_collected_with_df(collected_dfs, key, df)
                del df


def combine_recurse_results(results: list[dict[str, Any]]):
    """Combines the results of recursive operations into a single dictionary of DataFrames.

    The recursive returns may have been uris or explicit results. This function combines them into a single dictionary
    of DataFrames.

    Args:
      results: A list of dictionaries representing the results of recursive operations.

    Returns:
      collected_dfs: A dictionary containing the combined DataFrames, where the keys are the URIs of the DataFrames.

    Raises:
      CombineRecurseResultsMultipleKeysError: If a result dictionary contains more than one key when it has a "uri" key.
    """
    collected_dfs: dict[str, pd.DataFrame] = {}

    for result in results:
        if "uri" not in result:
            handle_explicit_result(collected_dfs, result)
        else:
            if len(result) > 1:
                raise CombineRecurseResultsMultipleKeysError
            uri = result["uri"]
            handle_referenced_result(collected_dfs, uri)
    return collected_dfs


class CombineRecurseResultsMultipleKeysError(ValueError):
    """Raised when a result dictionary contains more than one key when it has a "uri" key."""

    def __init__(self):
        """Initializes the error."""
        super().__init__("Cannot have both a uri and other keys in the results dict when combining recurse results")
