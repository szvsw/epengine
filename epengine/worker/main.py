import os

from pydantic_settings import BaseSettings

from epengine.hatchet import hatchet
from epengine.workflows import Fanout, Simulate


class SimWorkerSettings(BaseSettings):
    sim_worker_name_suffix: str = "cloud"
    FLY_REGION: str | None = None


def run():
    settings = SimWorkerSettings()
    cpu_ct = os.cpu_count() or 1
    max_runs = max(1, cpu_ct - 1)

    worker = hatchet.worker(
        f"ep-worker-{max_runs:03d}mr-{settings.sim_worker_name_suffix}",
        max_runs=max_runs,
    )

    if settings.FLY_REGION == "sea" or settings.FLY_REGION is None:
        worker.register_workflow(Fanout())

    worker.register_workflow(Simulate())

    worker.start()


if __name__ == "__main__":
    run()
