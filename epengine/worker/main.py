import os

from pydantic_settings import BaseSettings

from epengine.hatchet import hatchet
from epengine.workflows import ScatterGatherRecursiveWorkflow, ScatterGatherWorkflow, Simulate


class SimWorkerSettings(BaseSettings):
    FLY_REGION: str | None = None
    AWS_BATCH_JOB_ARRAY_INDEX: int | None = None

    @property
    def in_aws(self) -> bool:
        return self.AWS_BATCH_JOB_ARRAY_INDEX is not None

    @property
    def in_fly(self) -> bool:
        return self.FLY_REGION is not None

    @property
    def name(self) -> str:
        base = "EnergyPlusWorker"
        hosting = "AWS" if self.in_aws else ("Fly" if self.in_fly else "Local")
        hosting_with_region = f"{hosting}{self.FLY_REGION.upper()}" if self.FLY_REGION else hosting
        max_runs = self.max_runs
        return f"{base}--{hosting_with_region}--{max_runs:03d}slots"

    @property
    def max_runs(self) -> int:
        cpu_ct = os.cpu_count() or 1
        return max(1, cpu_ct - 1)


def run():
    settings = SimWorkerSettings()
    max_runs = settings.max_runs

    worker = hatchet.worker(
        settings.name,
        max_runs=max_runs,
    )

    if settings.FLY_REGION == "sea" or settings.FLY_REGION is None:
        worker.register_workflow(ScatterGatherWorkflow())
        worker.register_workflow(ScatterGatherRecursiveWorkflow())

    worker.register_workflow(Simulate())

    worker.start()


if __name__ == "__main__":
    run()
