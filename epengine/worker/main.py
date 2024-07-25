from epengine.hatchet import hatchet
from epengine.workflows import Fanout, Simulate


def run():
    worker = hatchet.worker(
        "ep-worker",
        max_runs=4,
    )

    # TODO: use two separate workers for this?
    # TODO: should we be using the async worker registration?
    worker.register_workflow(Simulate())
    worker.register_workflow(Fanout())

    worker.start()


if __name__ == "__main__":
    run()
