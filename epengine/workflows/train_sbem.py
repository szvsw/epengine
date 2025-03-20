"""Workflows for iteratively issuing simulations and training a regressor."""

import asyncio
from typing import cast

import boto3
from hatchet_sdk import Context

from epengine.hatchet import hatchet
from epengine.models.outputs import URIResponse
from epengine.models.train_sbem import (
    ConvergenceThresholds,
    CrossValidationSpec,
    IterationSpec,
    ProgressiveTrainingSpec,
    SampleSpec,
    StratificationSpec,
    TrainFoldSpec,
    TrainWithCVSpec,
)
from epengine.utils.results import serialize_df_dict

# TODO: this shared client should probably have better singleton config
s3 = boto3.client("s3")

# TODO: implement init


# TODO: implement sample + simualte workflow
@hatchet.workflow(
    name="sample_and_simulate",
    timeout="1000m",
    schedule_timeout="10m",
    version="0.1",
)
class SampleAndSimulate:
    """This workflow will sample the training data and simulate the results.

    It proceeds through the following states:
    - sample - from the GIS data, create a sample to simulate using stratification.
    - await_simulations - wait for the simulations to complete.
    - combine_results - combine the results of this round of simulations with the results from a previous stage.
    - state_transition - transition to training.
    """

    @hatchet.step(name="allocate", timeout="10m")
    async def allocate(self, context: Context):
        """This step is responsible for sampling the training data."""
        workflow_input = context.workflow_input()
        sample_spec = SampleSpec(**workflow_input)
        context.log("Sampling and uploading payload to s3...")
        payload = await asyncio.to_thread(sample_spec.make_payload, s3_client=s3)
        context.log("Payload uploaded to s3.")
        workflow_ref = await context.aio.spawn_workflow(
            workflow_name="scatter_gather_recursive",
            input=payload,
        )
        return {"id": workflow_ref.workflow_run_id}

    @hatchet.step(name="await_simulations", timeout="1000m", parents=["allocate"])
    async def await_simulations(self, context: Context):
        """This step is responsible for awaiting the simulations."""
        workflow_id = cast(dict[str, str], context.step_output("allocate"))["id"]
        workflow_ref = context.admin_client.get_workflow_run(workflow_id)
        context.log("Awaiting simulations...")
        result = await workflow_ref.result()
        context.log("Simulations completed.")
        uri_response = URIResponse(**result["collect_children"])
        return uri_response.model_dump(mode="json")

    @hatchet.step(name="combine_results", timeout="20m", parents=["await_simulations"])
    async def combine_results(self, context: Context):
        """This step is responsible for combining the results of the simulations."""
        workflow_input = context.workflow_input()
        sample_spec = SampleSpec(**workflow_input)
        step_output = cast(dict[str, str], context.step_output("await_simulations"))
        new_data_uri = URIResponse.model_validate(step_output)
        new_uri = await asyncio.to_thread(
            sample_spec.combine_results, new_data_uri=new_data_uri, s3_client=s3
        )
        return URIResponse.model_validate({"uri": new_uri}).model_dump(mode="json")

    @hatchet.step(name="state_transition", timeout="2m", parents=["combine_results"])
    async def state_transition(self, context: Context):
        """This step is responsible for transitioning the state of the workflow to training."""
        workflow_input = context.workflow_input()
        sample_spec = SampleSpec(**workflow_input)
        data_uri = cast(dict[str, str], context.step_output("combine_results"))["uri"]
        train_spec = TrainWithCVSpec(
            progressive_training_spec=sample_spec.progressive_training_spec,
            progressive_training_iteration_ix=sample_spec.progressive_training_iteration_ix,
            data_uri=data_uri,  # pyright: ignore [reportArgumentType]
            stage_type="train",
        )
        workflow_ref = await context.aio.spawn_workflow(
            workflow_name="train_regressor_with_cv",
            input=train_spec.model_dump(mode="json"),
        )
        return {"action": "train", "id": workflow_ref.workflow_run_id}


# TODO: implement state transition from sample + simulate to train


@hatchet.workflow(
    name="train_regressor_with_cv",
    timeout="10m",
    version="0.1",
)
class TrainRegressorWithCV:
    """This workflow will launch a scatter gather task to train each fold in a k-fold cross-validation."""

    @hatchet.step(name="allocate", timeout="10m")
    async def allocate(self, context: Context):
        """This step is responsible for launching the scatter gather task for kfold cross-validation."""
        workflow_input = context.workflow_input()
        train_spec = TrainWithCVSpec(**workflow_input)
        context.log("Creating training kfold cross-validation payload...")
        payload = await asyncio.to_thread(train_spec.allocate, s3_client=s3)
        context.log("Training kfold cross-validation payload created.")
        workflow_ref = await context.aio.spawn_workflow(
            workflow_name="scatter_gather",
            input=payload,
        )
        return {"id": workflow_ref.workflow_run_id}

    @hatchet.step(name="await_results", timeout="10m", parents=["allocate"])
    async def await_results(self, context: Context):
        """This step is responsible for awaiting the results of the scatter gather task."""
        workflow_id = cast(dict[str, str], context.step_output("allocate"))["id"]
        context.log("Awaiting results of scatter gather kfold cross-validation task...")
        workflow_ref = context.admin_client.get_workflow_run(workflow_id)
        context.log("Scatter gather kfold cross-validation task completed.")
        result = await workflow_ref.result()
        if "collect_children" not in result:
            msg = "Scatter gather kfold cross-validation task did not complete."
            raise ValueError(msg)
        uri_response = URIResponse.model_validate(result["collect_children"])
        return uri_response.model_dump(mode="json")

    @hatchet.step(name="check_convergence", timeout="20m", parents=["await_results"])
    async def check_convergence(self, context: Context):
        """This step is responsible for checking the convergence of the training."""
        workflow_input = context.workflow_input()
        train_spec = TrainWithCVSpec(**workflow_input)
        step_output = cast(dict[str, str], context.step_output("await_results"))
        uri_response = URIResponse.model_validate(step_output)
        # TODO: log convergence
        convergence_all, _convergence = await asyncio.to_thread(
            train_spec.check_convergence, uri=uri_response, s3_client=s3
        )

        return {"converged": bool(convergence_all)}

    @hatchet.step(name="state_transition", timeout="1m", parents=["check_convergence"])
    async def state_transition(self, context: Context):
        """Handle state transition logic."""
        workflow_input = context.workflow_input()
        train_spec = TrainWithCVSpec(**workflow_input)
        converged = cast(dict[str, bool], context.step_output("check_convergence"))[
            "converged"
        ]
        if converged:
            # go to cleanup
            return {
                "converged": True,
                "termination_reason": "converged",
                "action": "terminate",
            }

        if (
            train_spec.progressive_training_iteration_ix + 1
        ) >= train_spec.progressive_training_spec.iteration.max_iters:
            return {
                "converged": False,
                "termination_reason": "max_iters",
                "action": "terminate",
            }

        # go to sampler
        sample_spec = SampleSpec(
            progressive_training_spec=train_spec.progressive_training_spec,
            progressive_training_iteration_ix=train_spec.progressive_training_iteration_ix
            + 1,
            stage_type="sample",
            data_uri=train_spec.data_uri,
        )
        workflow_ref = await context.aio.spawn_workflow(
            workflow_name="sample_and_simulate",
            input=sample_spec.model_dump(mode="json"),
        )
        return {
            "converged": False,
            "id": workflow_ref.workflow_run_id,
            "action": "sample_and_simulate",
        }


@hatchet.workflow(
    name="train_regressor_with_cv_fold",
    timeout="10m",
    version="0.1",
)
class TrainRegressorWithCVFold:
    """This workflow will train a single fold of a k-fold cross-validation."""

    # TODO: we shouldn't have to name this step simulate for the scatter gather to work.
    @hatchet.step(name="simulate", timeout="10m")
    async def simulate(self, context: Context):
        """This step is responsible for training the model on a single fold."""

        def run():
            workflow_input = context.workflow_input()
            train_spec = TrainFoldSpec(**workflow_input)
            return serialize_df_dict(train_spec.run())

        return await asyncio.to_thread(run)


if __name__ == "__main__":
    from hatchet_sdk import new_client

    client = new_client()

    # input_gis_file = "./artifacts/prog-train-specs-with-two-regions.pq"
    # input_component_map_file = "./artifacts/component-map.yml"
    # input_semantic_fields_file = "./artifacts/semantic-fields.yml"
    # input_database_file = "./artifacts/components.db"
    input_gis_file = "./artifacts/ma-geometry.pq"
    input_component_map_file = "./artifacts/component-map-ma-simple.yml"
    input_semantic_fields_file = "./artifacts/semantic-fields-ma-simple.yml"
    input_database_file = "./artifacts/components-ma-simple.db"
    experiment_id = "test/progressive-training-22"
    bucket = "ml-for-bem"
    bucket_prefix = "hatchet"
    existing_artifacts = "forbid"

    # check that the experiment does not yet exist
    if (
        existing_artifacts == "forbid"
        and s3.list_objects_v2(
            Bucket=bucket, Prefix=f"{bucket_prefix}/{experiment_id}"
        ).get("KeyCount", 0)
        > 0
    ):
        msg = f"Experiment {experiment_id} already exists."
        raise ValueError(msg)

    # upload the input gis file
    gis_key = f"{bucket_prefix}/{experiment_id}/artifacts/gis.pq"
    gis_uri = f"s3://{bucket}/{gis_key}"
    s3.upload_file(input_gis_file, bucket, gis_key)

    # upload the input component map file
    component_map_key = f"{bucket_prefix}/{experiment_id}/artifacts/component-map.yml"
    component_map_uri = f"s3://{bucket}/{component_map_key}"
    s3.upload_file(input_component_map_file, bucket, component_map_key)

    # upload the input semantic fields file
    semantic_fields_key = (
        f"{bucket_prefix}/{experiment_id}/artifacts/semantic-fields.yml"
    )
    semantic_fields_uri = f"s3://{bucket}/{semantic_fields_key}"
    s3.upload_file(input_semantic_fields_file, bucket, semantic_fields_key)

    # upload the input database file
    database_key = f"{bucket_prefix}/{experiment_id}/artifacts/components.db"
    database_uri = f"s3://{bucket}/{database_key}"
    s3.upload_file(input_database_file, bucket, database_key)

    progressive_training_spec = ProgressiveTrainingSpec(
        iteration=IterationSpec(
            max_iters=10,
            max_samples=10000,  # TODO: this is currently unused
            n_per_iter=5000,
            n_init=5000,
            recursion_factor=7,
            recursion_max_depth=1,
            min_per_stratum=100,
        ),
        convergence_criteria=ConvergenceThresholds(
            mae=3,
            rmse=5,
            mape=0.05,
            r2=0.95,
            cvrmse=0.05,
        ),
        bucket=bucket,
        experiment_id=experiment_id,
        stratification=StratificationSpec(
            field="feature.weather.file",
            sampling="equal",
            aliases=["epwzip_path", "epwzip_uri"],
        ),
        cross_val=CrossValidationSpec(
            n_folds=5,
        ),
        gis_uri=gis_uri,  # pyright: ignore [reportArgumentType]
        component_map_uri=component_map_uri,  # pyright: ignore [reportArgumentType]
        semantic_fields_uri=semantic_fields_uri,  # pyright: ignore [reportArgumentType]
        database_uri=database_uri,  # pyright: ignore [reportArgumentType]
    )

    sample_spec = SampleSpec(
        progressive_training_spec=progressive_training_spec,
        progressive_training_iteration_ix=0,
        stage_type="sample",
        data_uri=None,
    )
    client.admin.run_workflow(
        workflow_name="sample_and_simulate",
        input=sample_spec.model_dump(mode="json"),
    )

    # train_spec = TrainWithCVSpec(
    #     progressive_training_spec=progressive_training_spec,
    #     progressive_training_iteration_ix=20,
    #     data_uri="s3://ml-for-bem/hatchet/braga-baseline-test-22/results/dataset-training.pq",  # pyright: ignore [reportArgumentType]
    #     stage_type="train",
    # )

    # client = new_client()

    # client.admin.run_workflow(
    #     workflow_name="train_regressor_with_cv",
    #     input=train_spec.model_dump(mode="json"),
    # )
