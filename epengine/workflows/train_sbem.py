"""Workflows for iteratively issuing simulations and training a regressor."""

import asyncio
from typing import cast

import boto3
from hatchet_sdk import Context

from epengine.hatchet import hatchet
from epengine.models.train_sbem import TrainFoldSpec, TrainWithCVSpec
from epengine.utils.results import serialize_df_dict

# TODO: this shared client should probably have better singleton config
s3 = boto3.client("s3")

# TODO: implement init

# TODO: implement sample + simualte workflow

# TODO: implement state transition from sample + simulate to train

# TODO: implement state transition from train to either terminate or sample + simulate


@hatchet.workflow(
    name="train_regressor_with_cv",
    timeout="10m",
    version="0.1",
)
class TrainRegressorWithCV:
    """This workflow will launch a scatter gather task to train each fold in a k-fold cross-validation."""

    @hatchet.step(name="allocate", timeout="10m")
    async def train(self, context: Context):
        """This step is responsible for launching the scatter gather task and returning the convergence results."""
        workflow_input = context.workflow_input()
        train_spec = TrainWithCVSpec(**workflow_input)
        return await train_spec.allocate(context=context, s3_client=s3)

    @hatchet.step(name="state_transition", timeout="20m", parents=["allocate"])
    async def state_transition(self, context: Context):
        """This step is responsible for transitioning the state of the workflow."""
        workflow_input = context.workflow_input()
        train_spec = TrainWithCVSpec(**workflow_input)
        workflow_id = cast(dict[str, str], context.step_output("allocate"))["id"]
        workflow_ref = context.admin_client.get_workflow_run(workflow_id)
        return await train_spec.state_transition(
            context=context, workflow_ref=workflow_ref
        )


# @hatchet.workflow(
#     name="train_regressor_with_cv",
#     timeout="10m",
#     version="0.1",
# )
# class TrainRegressorWithCV:
#     """This workflow will launch a scatter gather task to train each fold in a k-fold cross-validation."""

#     @hatchet.step(name="allocate", timeout="10m")
#     async def train(self, context: Context):
#         """This step is responsible for launching the scatter gather task and returning the convergence results."""
#         workflow_input = context.workflow_input()
#         train_spec = TrainWithCVSpec(**workflow_input)
#         return await train_spec.allocate(context=context, s3_client=s3)


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
    n_folds = 3
    train_spec = TrainWithCVSpec(
        bucket="ml-for-bem",
        experiment_id="test-train-cv",
        n_folds=n_folds,
        data_uri="s3://ml-for-bem/hatchet/braga-baseline-test-22/results/dataset-training.pq",  # pyright: ignore [reportArgumentType]
        stratification_field="feature.weather.file",
        progressive_training_iter_ix=0,
    )
    from hatchet_sdk.client import new_client

    client = new_client()

    client.admin.run_workflow(
        workflow_name="train_regressor_with_cv",
        input=train_spec.model_dump(mode="json"),
    )
