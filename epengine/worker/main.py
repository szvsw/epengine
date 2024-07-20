from hatchet_sdk import Hatchet

from epengine.workflows import MyWorkflow

hatchet = Hatchet()

worker = hatchet.worker("first-worker")
worker.register_workflow(MyWorkflow())

worker.start()
