"""Main entrypoint for the EnergyPlus worker."""

import os

from pydantic_settings import BaseSettings

from epengine.hatchet import hatchet
from epengine.workflows import (
    ScatterGatherRecursiveWorkflow,
    ScatterGatherWorkflow,
    Simulate,
)


class SimWorkerSettings(BaseSettings):
    """Settings for the EnergyPlus worker."""

    FLY_REGION: str | None = None
    AWS_BATCH_JOB_ARRAY_INDEX: int | None = None

    @property
    def in_aws(self) -> bool:
        """Return whether the worker is running in AWS Batch."""
        return self.AWS_BATCH_JOB_ARRAY_INDEX is not None

    @property
    def in_fly(self) -> bool:
        """Return whether the worker is running in Fly.io."""
        return self.FLY_REGION is not None

    @property
    def name(self) -> str:
        """Return the name of the worker."""
        base = "EnergyPlusWorker"
        hosting = "AWS" if self.in_aws else ("Fly" if self.in_fly else "Local")
        hosting_with_region = (
            f"{hosting}{self.FLY_REGION.upper()}" if self.FLY_REGION else hosting
        )
        max_runs = self.max_runs
        return f"{base}--{hosting_with_region}--{max_runs:03d}slots"

    @property
    def max_runs(self) -> int:
        """Return the maximum number of runs."""
        cpu_ct = os.cpu_count() or 1
        if cpu_ct < 8:
            return cpu_ct
        else:
            return cpu_ct - 1

    def make_worker(self):
        """Create a worker with the settings.

        Handles registering the appropriate workflows based on the settings.

        Returns:
            Worker: The worker.

        """
        worker = hatchet.worker(
            self.name,
            max_runs=self.max_runs,
        )

        if self.FLY_REGION == "sea" or self.FLY_REGION is None:
            worker.register_workflow(ScatterGatherWorkflow())
            worker.register_workflow(ScatterGatherRecursiveWorkflow())

        worker.register_workflow(Simulate())

        return worker


async def arun():
    """Run the EnergyPlus worker.

    Note that this function will be blocking.
    """
    settings = SimWorkerSettings()
    worker = settings.make_worker()

    await worker.async_start()


def run():
    """Run the EnergyPlus worker.

    Note that this function will be blocking.
    """
    settings = SimWorkerSettings()
    worker = settings.make_worker()
    worker.start()


if __name__ == "__main__":
    run()
