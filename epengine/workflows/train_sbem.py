"""Workflows for iteratively issuing simulations and training a regressor."""

import boto3
from hatchet_sdk import Context, sync_to_async

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

    @hatchet.step(name="train", timeout="10m")
    async def train(self, context: Context):
        """This step is responsible for launching the scatter gather task and returning the convergence results."""
        return await safe_cv(context)


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
        return await safe_train(context)


@sync_to_async
async def safe_cv(context: Context):
    """This function is responsible for allocating the train with cv spec and returning the convergence results with a wrapper for async safety."""
    workflow_input = context.workflow_input()
    train_spec = TrainWithCVSpec(**workflow_input)
    return await train_spec.allocate(context=context, s3_client=s3)


@sync_to_async
def safe_train(context: Context):
    """This function is responsible for training the model on a single fold and returning the results with a wrapper for async safety."""
    workflow_input = context.workflow_input()
    train_spec = TrainFoldSpec(**workflow_input)
    return serialize_df_dict(train_spec.run())
