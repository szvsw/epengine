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


if __name__ == "__main__":
    from hatchet_sdk import new_client

    client = new_client()
    boston_lat = 42.3601
    boston_lon = -71.0589
    spec = SBEMInferenceRequestSpec(
        lat=boston_lat,
        lon=boston_lon,
        rotated_rectangle="POLYGON ((5 0, 5 10, 15 10, 15 0, 5 0))",
        neighbor_polys=["POLYGON ((-10 0, -10 10, -5 10, -5 0, -10 0))"],
        neighbor_floors=[3],
        short_edge=5,
        long_edge=10,
        num_floors=3,
        orientation=0.001,
        basement="none",
        attic="none",
        semantic_field_context={
            "Age_bracket": "btw_1975_2003",
            "AtticFloorInsulation": "Insulated",
            "AtticVentilation": "VentilatedAttic",
            "BasementCeilingInsulation": "UninsulatedCeiling",
            "BasementWallsInsulation": "InsulatedWalls",
            "Cooling": "ACWindow",
            "DHW": "HPWH",
            "Distribution": "AirDuctsConditionedUninsulated",
            "Equipment": "HighEfficiencyEquipment",
            "GroundSlabInsulation": "UninsulatedGroundSlab",
            "Heating": "NaturalGasCondensingHeating",
            "Lighting": "LED",
            "Region": "MA",
            "RoofInsulation": "InsulatedRoof",
            "Thermostat": "NoControls",
            "Typology": "MFH",
            "Walls": "FullInsulationWallsCavityExterior",
            "Weatherization": "LeakyEnvelope",
            "Windows": "SinglePane",
        },
        source_experiment="ma-webapp/v0-20250407-171250",
        actual_conditioned_area_m2=100,
    )

    client.admin.run_workflow(
        "sbem_inference",
        spec.model_dump(mode="json"),
    )
