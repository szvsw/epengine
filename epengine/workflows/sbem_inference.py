"""Workflow for running inference for a SBEM model."""

import asyncio

from hatchet_sdk import Context

from epengine.hatchet import hatchet
from epengine.models.inference import SBEMInferenceRequestSpec


@hatchet.workflow(
    name="sbem_inference",
    version="1.0.0",
    timeout="30s",
    schedule_timeout="30s",
)
class SBEMInferenceWorkflow:
    """A workflow for running inference for a SBEM model."""

    @hatchet.step(
        name="simulate",
        timeout="30s",
    )
    async def simulate(self, context: Context):
        """Simulate the SBEM model."""
        spec = SBEMInferenceRequestSpec(**context.workflow_input())

        results = await asyncio.to_thread(spec.run)

        return results.serialized.model_dump(mode="json")
