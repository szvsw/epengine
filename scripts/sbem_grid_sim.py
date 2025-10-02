"""This script is used to run an sbem grid sim for basic sensitivity analysis."""

from typing import cast

import pandas as pd
from pydantic import AnyUrl
from tqdm import tqdm

from epengine.models.shoebox_sbem import SBEMSimulationSpec


def run_with_semantic_envelopes(
    wwr: float,
    num_floors: int,
    roof: str,
    walls: str,
    windows: str,
    weatherization: str,
):
    """Run an sbem grid sim for basic sensitivity analysis."""
    ## TODO: remember to update the db_uri and semantic_fields_uri etc
    spec = SBEMSimulationSpec(
        experiment_id="test",
        sort_index=0,
        rotated_rectangle="POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))",
        num_floors=num_floors,
        height=num_floors * 2.8,
        long_edge=10,
        short_edge=10,
        aspect_ratio=1,
        rotated_rectangle_area_ratio=1,
        long_edge_angle=0.0,
        wwr=wwr,
        f2f_height=2.8,
        neighbor_polys=[],
        neighbor_floors=[],
        neighbor_heights=[],
        epwzip_uri=AnyUrl(
            "https://climate.onebuilding.org/WMO_Region_4_North_and_Central_America/USA_United_States_of_America/MA_Massachusetts/USA_MA_Boston-Logan.Intl.AP.725090_TMYx.2009-2023.zip"
        ),
        db_uri=AnyUrl(
            "s3://ml-for-bem/hatchet/ma-webapp/test/v2-20250508-122929-boston-only/artifacts/components-ma.db"
        ),
        semantic_fields_uri=AnyUrl(
            "s3://ml-for-bem/hatchet/ma-webapp/test/v2-20250508-122929-boston-only/artifacts/semantic-fields.yml"
        ),
        component_map_uri=AnyUrl(
            "s3://ml-for-bem/hatchet/ma-webapp/test/v2-20250508-122929-boston-only/artifacts/component-map.yml"
        ),
        semantic_field_context={
            "AtticVentilation": "UnventilatedAttic",
            "BasementCeilingInsulation": "UninsulatedCeiling",
            "BasementWallsInsulation": "UninsulatedWalls",
            "GroundSlabInsulation": "UninsulatedGroundSlab",
            "AtticFloorInsulation": "NoInsulation",
            "Region": "MA",
            "Typology": "SFH",
            "Age_bracket": "pre_1975",
            "Heating": "NaturalGasHeating",
            "Cooling": "ACWindow",
            "Distribution": "HotWaterUninsulated",
            "DHW": "NaturalGasDHW",
            "Lighting": "LED",
            "Thermostat": "NoControls",
            "Equipment": "LowEfficiencyEquipment",
            # these are the ones that we will be changing
            "RoofInsulation": roof,
            "Walls": walls,
            "Windows": windows,
            # "Weatherization": "SomewhatLeakyEnvelope",
            "Weatherization": weatherization,
        },
        basement="none",
        attic="none",
        exposed_basement_frac=0.25,
    )
    _, res, _ = spec.run()

    raw: pd.DataFrame = cast(pd.DataFrame, res["Energy"]["Raw"])
    return raw.T.groupby("Meter").sum().T / 3.154


if __name__ == "__main__":
    roofs = [
        "UninsulatedRoof",
        "InsulatedRoof",
        "HighlyInsulatedRoof",
    ]

    walls = [
        "SomeInsulationWalls",
        "FullInsulationWallsCavity",
        "FullInsulationWallsCavityExterior",
    ]

    windows = [
        "SinglePane",
        "DoublePaneLowE",
        "TriplePaneLowE",
    ]

    weatherization = [
        "LeakyEnvelope",
        "SomewhatLeakyEnvelope",
        "TightEnvelope",
    ]

    wwr = 0.12
    num_floors = 8

    used_roofs = []
    used_walls = []
    used_windows = []
    used_weatherization = []
    heating = []
    cooling = []

    for roof in tqdm(roofs):
        for wall in tqdm(walls):
            for window in windows:
                for w in weatherization:
                    res = run_with_semantic_envelopes(
                        wwr, num_floors, roof, wall, window, w
                    )
                    used_roofs.append(roof)
                    used_walls.append(wall)
                    used_windows.append(window)
                    used_weatherization.append(w)
                    heating.append(res.Heating.values[0])
                    cooling.append(res.Cooling.values[0])

    df = pd.DataFrame({
        "Roof": used_roofs,
        "Wall": used_walls,
        "Window": used_windows,
        "Weatherization": used_weatherization,
        "Heating": heating,
        "Cooling": cooling,
    })
