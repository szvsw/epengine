"""Main entrypoint for the EnergyPlus worker."""

import os

from pydantic_settings import BaseSettings

from epengine.hatchet import hatchet
from epengine.workflows import (
    ScatterGatherRecursiveWorkflow,
    ScatterGatherWorkflow,
    SimpleTest,
    Simulate,
    SimulateSBEMShoebox,
    SimulateShoebox,
)


class SimWorkerSettings(BaseSettings):
    """Settings for the EnergyPlus worker."""

    FLY_REGION: str | None = None
    AWS_BATCH_JOB_ARRAY_INDEX: int | None = None
    COPILOT_ENVIRONMENT_NAME: str | None = None
    DOES_FAN: bool = True
    DOES_LEAF: bool = True
    MAX_RUNS: int | None = None

    @property
    def in_aws_batch(self) -> bool:
        """Return whether the worker is running in AWS Batch."""
        return self.AWS_BATCH_JOB_ARRAY_INDEX is not None

    @property
    def in_aws_copilot(self) -> bool:
        """Return whether the worker is running in AWS Copilot."""
        return self.COPILOT_ENVIRONMENT_NAME is not None

    @property
    def in_aws(self) -> bool:
        """Return whether the worker is running in AWS."""
        return self.in_aws_batch or self.in_aws_copilot

    @property
    def aws_hosting_str(self):
        """Return the AWS hosting string for the worker."""
        batch = (
            f"Batch{self.AWS_BATCH_JOB_ARRAY_INDEX:04d}" if self.in_aws_batch else ""
        )
        copilot = (
            f"Copilot{(self.COPILOT_ENVIRONMENT_NAME or '').upper()}"
            if self.in_aws_copilot
            else ""
        )
        return f"AWS{batch}{copilot}" if self.in_aws else ""

    @property
    def in_fly(self) -> bool:
        """Return whether the worker is running in Fly.io."""
        return self.FLY_REGION is not None

    @property
    def fly_hosting_str(self):
        """Return the Fly hosting string for the worker."""
        return f"Fly{(self.FLY_REGION or '').upper()}" if self.in_fly else ""

    @property
    def in_local(self) -> bool:
        """Return whether the worker is running locally."""
        return not self.in_aws and not self.in_fly

    @property
    def hosting_str(self):
        """Return the hosting string for the worker."""
        return self.aws_hosting_str or self.fly_hosting_str or "Local"

    @property
    def name(self) -> str:
        """Return the name of the worker."""
        base = "EnergyPlusWorker"
        max_runs = self.max_runs
        return f"{base}--{self.hosting_str}--{max_runs:03d}slots"

    @property
    def max_runs(self) -> int:
        """Return the maximum number of runs."""
        if self.MAX_RUNS is not None:
            return self.MAX_RUNS
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

        if (self.FLY_REGION == "sea" or self.FLY_REGION is None) and self.DOES_FAN:
            worker.register_workflow(ScatterGatherWorkflow())
            worker.register_workflow(ScatterGatherRecursiveWorkflow())

        if self.DOES_LEAF:
            worker.register_workflow(Simulate())
            worker.register_workflow(SimulateShoebox())
            worker.register_workflow(SimulateSBEMShoebox())
        worker.register_workflow(SimpleTest())

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
