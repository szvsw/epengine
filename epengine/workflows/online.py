"""A module for iteratively running simulations and training models until convergence.

The basic scheme of the workflow is as follows:

- run simulations and add them to the cache
- train a model on the cache
- check for convergence
- if converged, finalize the model
- if not converged, grow the cache again and train a new model

Init → [Grow Cache → Train Model → Check Convergence] → Finalize
        ↑__________________________________________|
                (loop until converged)
"""

from hatchet_sdk import Context

from epengine.hatchet import hatchet

"""
TODO/Notes

1. Use a seed to compress the train/test ix splits
2. Use recovered train/test ix splits to selectively read from columnar parquets (which is transposed so that the building ix is the column).
4. Knowledge of the stratifier(s) must be maintained (e.g. `epwzip_path` will be used for equally weighhted sampling)
5. Convergence criterion checked for all strata; these can actually be run in parallel.

Min samples per stratum
Max simulations or generation
Adding sims where not converged vs adding sims equally

"""


@hatchet.workflow(
    name="init_engine",
    version="1.0.0",
)
class InitEngine:
    """A workflow for initializing a model which will be trained online.

    This workflow is responsible for setting up resources in s3 buckets and configuring
    inputs that will be used in other stages of the training workflow, then entering
    the loop.

    """

    @hatchet.step(name="entrypoint")
    def entrypoint(self, context: Context):
        """This step is the entrypoint of the workflow.

        This step is responsible for setting up resources in s3 buckets and configuring
        inputs that will be used in other stages of the training workflow.
        """
        pass

    @hatchet.step(name="begin_loop", parents=["entrypoint"])
    async def begin_loop(self, context: Context):
        """This step is responsible for moving to the next phase of the workflow."""
        await context.aio.spawn_workflow("grow_cache", {})


@hatchet.workflow(
    name="grow_cache",
    version="1.0.0",
)
class GrowCache:
    """This workflow is responsible for growing the cache of training data.

    It will handle enqueuing simulations and gathering the results.
    """

    @hatchet.step(name="scatter")
    def scatter(self, context: Context):
        """This step is responsible for enqueuing simulations."""
        pass

    @hatchet.step(name="gather", parents=["scatter"])
    def gather(self, context: Context):
        """This step is responsible for gathering the results of the simulations."""
        pass

    @hatchet.step(name="launch_training", parents=["gather"])
    async def launch_training(self, context: Context):
        """This step is responsible for launching the training of a new model.

        It transitions to a separate workflow.
        """
        await context.aio.spawn_workflow("train_model", {})


@hatchet.workflow(
    name="train_model",
    version="1.0.0",
)
class TrainModel:
    """This workflow is responsible for training a model.

    It will handle the training of a model and checking for convergence, and then
    transition to the next phase of the workflow.
    """

    @hatchet.step(name="train")
    def train(self, context: Context):
        """This step is responsible for training a model based on the current cache."""
        pass

    @hatchet.step(name="check_convergence", parents=["train"])
    async def check_convergence(self, context: Context):
        """This step is responsible for checking for convergence of the model.

        If the model has not yet converged, it will recurse back to the cache-growing stage,
        otherwise it will exit to the model finalization workflow.
        """
        converged = False
        if converged:
            await context.aio.spawn_workflow("finalize_model", {})
        else:
            await context.aio.spawn_workflow("grow_cache", {})


@hatchet.workflow(
    name="finalize_model",
    version="1.0.0",
)
class FinalizeModel:
    """A workflow for finalizing a model."""

    @hatchet.step(name="finalize")
    def finalize(self, context: Context):
        """This step is responsible for finalizing the model.

        It will handle the finalization of any resources such as s3 buckets and
        model artifacts.
        """
        pass
