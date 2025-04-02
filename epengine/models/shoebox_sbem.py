"""A model for an SBEM composed shoebox simulation."""

import logging
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
import yaml
from archetypal.idfclass import IDF
from epinterface.geometry import (
    ShoeboxGeometry,
    compute_shading_mask,
    match_idf_to_building_and_neighbors,
)
from epinterface.sbem.builder import AtticAssumptions, BasementAssumptions, Model
from epinterface.sbem.components.composer import (
    construct_composer_model,
    construct_graph,
)
from epinterface.sbem.components.zones import ZoneComponent
from epinterface.sbem.prisma.client import PrismaSettings
from ladybug.epw import EPW
from pydantic import AnyUrl, Field

from epengine.models.base import LeafSpec

logger = logging.getLogger(__name__)


DAYS_PER_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
HOURS_PER_DAY = 24
HOURS_PER_MONTH = DAYS_PER_MONTH * HOURS_PER_DAY
MONTH_INDICES = np.arange(12).repeat(HOURS_PER_MONTH)


@dataclass
class EPWSummary:
    """A dataclass for storing EPW summary statistics."""

    annual_hdd: float
    annual_cdd: float

    annual_daily_median_median: float
    annual_daily_min_median: float
    annual_daily_max_median: float
    annual_daily_range_median: float

    monthly_hdd: np.ndarray
    monthly_cdd: np.ndarray

    monthly_daily_max: np.ndarray
    monthly_daily_min: np.ndarray

    monthly_daily_range_median: np.ndarray
    monthly_daily_median_median: np.ndarray
    monthly_daily_max_median: np.ndarray
    monthly_daily_min_median: np.ndarray

    latitude: float
    longitude: float

    @property
    def flat_dict(self):
        """Return a flattened dictionary of the EPW summary statistics."""
        data = {
            "feature.weather.annual.dd_heating": self.annual_hdd,
            "feature.weather.annual.dd_cooling": self.annual_cdd,
            "feature.weather.annual.daily_median_median": self.annual_daily_median_median,
            "feature.weather.annual.daily_min_median": self.annual_daily_min_median,
            "feature.weather.annual.daily_max_median": self.annual_daily_max_median,
            "feature.weather.annual.daily_range_median": self.annual_daily_range_median,
        }
        for i in range(12):
            data[f"feature.weather.monthly.dd_heating.{i + 1:02d}"] = self.monthly_hdd[
                i
            ]
            data[f"feature.weather.monthly.dd_cooling.{i + 1:02d}"] = self.monthly_cdd[
                i
            ]
            data[f"feature.weather.monthly.daily_min.{i + 1:02d}"] = (
                self.monthly_daily_min[i]
            )
            data[f"feature.weather.monthly.daily_max.{i + 1:02d}"] = (
                self.monthly_daily_max[i]
            )
            data[f"feature.weather.monthly.daily_range_median.{i + 1:02d}"] = (
                self.monthly_daily_range_median[i]
            )
            data[f"feature.weather.monthly.daily_median_median.{i + 1:02d}"] = (
                self.monthly_daily_median_median[i]
            )
            data[f"feature.weather.monthly.daily_max_median.{i + 1:02d}"] = (
                self.monthly_daily_max_median[i]
            )
            data[f"feature.weather.monthly.daily_min_median.{i + 1:02d}"] = (
                self.monthly_daily_min_median[i]
            )
        data["feature.weather.latitude"] = self.latitude
        data["feature.weather.longitude"] = self.longitude

        return dict(sorted(data.items()))

    @property
    def series(self):
        """Return a pandas Series of the EPW summary statistics."""
        return pd.Series(self.flat_dict).sort_index()

    @classmethod
    def FromEPW(
        cls,
        epw: EPW | Path,
        base_heating_temperature: float = 18,
        base_cooling_temperature: float = 23,
    ) -> "EPWSummary":
        """Create an EPWSummary from an EPW file or path to an EPW file.

        Args:
            epw (EPW | Path): The EPW file or path to an EPW file.
            base_heating_temperature (float): The base heating temperature for degree day computations..
            base_cooling_temperature (float): The base cooling temperature for degree day computations.

        Returns:
            summary (EPWSummary): The EPW summary statistics.
        """
        if isinstance(epw, Path):
            if epw.suffix == ".epw":
                epw = EPW(epw)
            elif epw.suffix == ".zip":
                with zipfile.ZipFile(epw, "r") as zip_ref:
                    # only extract the epw
                    file_name = epw.with_suffix(".epw")
                    zip_ref.extract(file_name.name, epw.parent)
                    epw = EPW(file_name)
            else:
                msg = f"Expected .epw or .zip file, got {epw.suffix}."
                raise ValueError(msg)

        dry_bulb_temperatures = np.array(epw.dry_bulb_temperature.values)

        is_heating = dry_bulb_temperatures < base_heating_temperature
        is_cooling = dry_bulb_temperatures > base_cooling_temperature
        annual_hdd = is_heating.sum() / 24
        annual_cdd = is_cooling.sum() / 24
        annual_daily_median_median = np.median(
            np.median(dry_bulb_temperatures.reshape(-1, 24), axis=1)
        )
        annual_daily_min_median = np.median(
            np.min(dry_bulb_temperatures.reshape(-1, 24), axis=1)
        )
        annual_daily_max_median = np.median(
            np.max(dry_bulb_temperatures.reshape(-1, 24), axis=1)
        )
        annual_daily_range_median = np.median(
            np.max(dry_bulb_temperatures.reshape(-1, 24), axis=1)
            - np.min(dry_bulb_temperatures.reshape(-1, 24), axis=1)
        )
        monthly_hdd = [
            (is_heating & (i == MONTH_INDICES)).sum() / 24 for i in range(12)
        ]
        monthly_cdd = [
            (is_cooling & (i == MONTH_INDICES)).sum() / 24 for i in range(12)
        ]
        monthly_daily_median_median = [
            np.median(
                np.median(
                    dry_bulb_temperatures[i == MONTH_INDICES].reshape(-1, 24), axis=1
                )
            )
            for i in range(12)
        ]
        monthly_daily_max = [
            np.max(dry_bulb_temperatures[i == MONTH_INDICES]) for i in range(12)
        ]
        monthly_daily_max_median = [
            np.median(
                np.max(
                    dry_bulb_temperatures[i == MONTH_INDICES].reshape(-1, 24), axis=1
                )
            )
            for i in range(12)
        ]
        monthly_daily_min = [
            np.min(dry_bulb_temperatures[i == MONTH_INDICES]) for i in range(12)
        ]
        monthly_daily_min_median = [
            np.median(
                np.min(
                    dry_bulb_temperatures[i == MONTH_INDICES].reshape(-1, 24), axis=1
                )
            )
            for i in range(12)
        ]
        monthly_daily_range_median = [
            np.median(
                np.max(
                    dry_bulb_temperatures[i == MONTH_INDICES].reshape(-1, 24), axis=1
                )
                - np.min(
                    dry_bulb_temperatures[i == MONTH_INDICES].reshape(-1, 24), axis=1
                )
            )
            for i in range(12)
        ]
        latitude = epw.location.latitude
        longitude = epw.location.longitude
        return cls(
            annual_hdd=annual_hdd,
            annual_cdd=annual_cdd,
            annual_daily_median_median=annual_daily_median_median,
            annual_daily_min_median=annual_daily_min_median,
            annual_daily_max_median=annual_daily_max_median,
            annual_daily_range_median=annual_daily_range_median,
            monthly_hdd=np.array(monthly_hdd),
            monthly_cdd=np.array(monthly_cdd),
            monthly_daily_max=np.array(monthly_daily_max),
            monthly_daily_min=np.array(monthly_daily_min),
            monthly_daily_range_median=np.array(monthly_daily_range_median),
            monthly_daily_median_median=np.array(monthly_daily_median_median),
            monthly_daily_max_median=np.array(monthly_daily_max_median),
            monthly_daily_min_median=np.array(monthly_daily_min_median),
            latitude=latitude,
            longitude=longitude,
        )


class SBEMSimulationSpec(LeafSpec):
    """A spec for running an EnergyPlus simulation."""

    db_uri: AnyUrl = Field(
        ...,
        description="The uri of the db file to providing the building component library.",
    )
    semantic_fields_uri: AnyUrl = Field(
        ...,
        description="The uri of the semantic fields file representing the expected semantic fields.",
    )
    component_map_uri: AnyUrl = Field(
        ...,
        description="The uri of the component map file representing how to compile the zone definition from components given the semantic field context.",
    )
    epwzip_uri: AnyUrl = Field(
        ..., description="The uri of the epw file to use during simulation."
    )
    # assumptions: GisJobAssumptions = Field(
    #     ..., description="The assumptions for the simulation"
    # )
    semantic_field_context: dict[str, float | str | int] = Field(
        ...,
        description="The semantic field values which will be used to compile the zone definition.",
    )
    neighbor_polys: list[str] = Field(
        ..., description="The polygons of the neighboring buildings."
    )
    neighbor_heights: list[float | int | None] = Field(
        ..., description="The height of the neighboring buildings  [m]."
    )
    neighbor_floors: list[float | int | None] = Field(
        ..., description="The number of floors of the neighboring buildings."
    )
    rotated_rectangle: str = Field(
        ..., description="The rotated rectangle fitted around the base of the building."
    )
    long_edge_angle: float = Field(
        ..., description="The long edge angle of the building (radians)."
    )
    long_edge: float = Field(
        ..., description="The length of the long edge of the building [m]."
    )
    short_edge: float = Field(
        ..., description="The length of the short edge of the building [m]."
    )
    aspect_ratio: float = Field(
        ..., description="The aspect ratio of the building footprint [unitless]."
    )
    rotated_rectangle_area_ratio: float = Field(
        ...,
        description="The ratio of the rotated rectangle footprint area to the building footprint area.",
    )
    wwr: float = Field(
        ..., description="The window-to-wall ratio of the building [unitless]."
    )
    height: float = Field(..., description="The height of the building [m].")
    num_floors: int = Field(..., description="The number of floors in the building.")
    f2f_height: float = Field(..., description="The floor to floor height [m].")
    # TODO: add fp area? gfa?
    # footprint_area: float = Field(
    #     ..., description="The footprint area of the building [m^2]."
    # )
    # TODO: add some fields about requests for specific results, e.g. heating/cooling, monthly, utilities vs raw etc?

    @property
    def feature_dict(self) -> dict[str, str | int | float]:
        """Return a dictionary of features which will be available to ML algos."""
        # Static features represent features which are constant for a given building and not subject to estimation, sampling, etc.

        features: dict[str, str | int | float] = {
            "feature.geometry.long_edge": self.long_edge,
            "feature.geometry.short_edge": self.short_edge,
            "feature.geometry.orientation": self.long_edge_angle,
            "feature.geometry.orientation.cos": np.cos(self.long_edge_angle),
            "feature.geometry.orientation.sin": np.sin(self.long_edge_angle),
            "feature.geometry.aspect_ratio": self.aspect_ratio,
            "feature.geometry.rotated_rectangle_area_ratio": self.rotated_rectangle_area_ratio,
            "feature.geometry.wwr": self.wwr,
            "feature.geometry.height": self.height,
            "feature.geometry.num_floors": self.num_floors,
            "feature.geometry.f2f_height": self.f2f_height,
            "feature.geometry.zoning": self.use_core_perim_zoning,
            # TODO: add gfa? or just fp area? or pull from model.geometry?
        }

        # TODO: consider passing in
        # neighbors directly to Model.geometry, letting model perform neighbor
        # insertion directly rather than via a callback,
        # and then let shading mask become a computed property of the model.geometry.
        shading_mask = compute_shading_mask(
            self.rotated_rectangle,
            neighbors=self.neighbor_polys,
            neighbor_heights=self.neighbor_heights,
            azimuthal_angle=2 * np.pi / 48,
        )
        shading_mask_values = {
            f"feature.geometry.shading_mask_{i:02d}": val
            for i, val in enumerate(shading_mask.tolist())
        }
        features.update(shading_mask_values)

        # semantic features are kept separately as one building may have
        # multiple simulations with different semantic fields.
        features.update({
            f"feature.semantic.{feature_name}": feature_value
            for feature_name, feature_value in self.semantic_field_context.items()
        })

        features["feature.weather.file"] = self.epwzip_path.stem
        features.update(EPWSummary.FromEPW(self.epwzip_path).flat_dict)

        # where to put ff2 height, wwr?

        # conditional features are derived from the static and semantic features,
        # and may be subject to things like conditional sampling, estimation etc.
        # e.g. rvalues, uvalues, schedule, etc.
        # additional things like basement/attic config?

        return features

    @cached_property
    def db_path(self):
        """Fetch the db file and return the local path.

        Returns:
            local_path (Path): The local path of the idf file
        """
        return self.fetch_uri(self.db_uri)

    @cached_property
    def semantic_fields_path(self):
        """Fetch the semantic fields file and return the local path.

        Returns:
            local_path (Path): The local path of the semantic fields file
        """
        return self.fetch_uri(self.semantic_fields_uri)

    @cached_property
    def epwzip_path(self):
        """Fetch the epw file and return the local path.

        Returns:
            local_path (Path): The local path of the epw file
        """
        return self.fetch_uri(self.epwzip_uri)

    @property
    def component_map(self):
        """Fetch the component map file and return the local path.

        Returns:
            local_path (Path): The local path of the component map file
        """
        return self.fetch_uri(self.component_map_uri)

    def construct_zone_def(self) -> ZoneComponent:
        """Construct the zone definition for the simulation.

        Returns:
            zone_def (ZoneComponent): The zone definition for the simulation
        """
        g = construct_graph(ZoneComponent)
        SelectorModel = construct_composer_model(
            g,
            ZoneComponent,
            use_children=False,
        )

        with open(self.component_map) as f:
            component_map_yaml = yaml.safe_load(f)
        selector = SelectorModel.model_validate(component_map_yaml)

        # TODO: make sure we are okay with accwssing the same db
        # across workers executing the same experiment.
        settings = PrismaSettings.New(
            database_path=self.db_path, if_exists="ignore", auto_register=False
        )
        db = settings.db

        context = self.semantic_field_context
        with db:
            return cast(ZoneComponent, selector.get_component(context=context, db=db))

    @property
    def use_core_perim_zoning(self) -> Literal["by_storey", "core/perim"]:
        """Whether to use the core perimeter for the simulation."""
        use_core_perim = self.long_edge > 15 and self.short_edge > 15
        return "core/perim" if use_core_perim else "by_storey"

    def run(self, log_fn: Callable | None = None):
        """Run the simulation.

        Args:
            log_fn (Callable | None): The function to use for logging.
        """
        log = log_fn or logger.info
        zone_def = self.construct_zone_def()

        # TODO: Geometry loading, weather loading, geometry post process, etc
        model = Model(
            Weather=self.epwzip_path,
            Zone=zone_def,
            # TODO: compute these somehow?
            Basement=BasementAssumptions(
                Conditioned=False,
                UseFraction=None,
            ),
            Attic=AtticAssumptions(
                Conditioned=False,
                UseFraction=None,
            ),
            geometry=ShoeboxGeometry(
                x=0,
                y=0,
                w=self.long_edge,
                d=self.short_edge,
                h=self.f2f_height,
                wwr=self.wwr,
                num_stories=self.num_floors,
                basement=False,
                zoning=self.use_core_perim_zoning,
                roof_height=None,
            ),
        )

        # TODO: implement post-build callback.
        # and maybe move it into epinterface.geometry
        # using neighboring context as argument to
        # model.geometry.
        def post_geometry_callback(idf: IDF) -> IDF:
            log("Matching IDF to building and neighbors...")
            idf = match_idf_to_building_and_neighbors(
                idf,
                building=self.rotated_rectangle,
                neighbor_polys=self.neighbor_polys,  # pyright: ignore [reportArgumentType]
                # neighbor_floors=[
                #     (floor / self.f2f_height) if floor is not None else None
                #     for floor in self.neighbor_floors
                # ],
                neighbor_floors=self.neighbor_floors,
                neighbor_f2f_height=self.f2f_height,
                target_short_length=self.short_edge,
                target_long_length=self.long_edge,
                rotation_angle=self.long_edge_angle,
            )
            log("IDF matched to building and neighbors.")
            return idf

        log("Building and running model...")
        idf, results, err_text = model.run(
            post_geometry_callback=post_geometry_callback
        )
        # creating results dataframe index

        dumped_self = self.model_dump(
            mode="json",
            exclude={
                "feature_dict",
                "neighbor_polys",
                "neighbor_floors",
                "neighbor_heights",
                "semantic_field_context",
            },
        )
        dumped_self.update(self.feature_dict)
        dumped_self["feature.geometry.total_conditioned_area"] = (
            model.total_conditioned_area
        )
        dumped_self["feature.extra_spaces.basement.occupied"] = (
            model.Basement.UseFraction or 0
        )
        dumped_self["feature.extra_spaces.basement.conditioned"] = (
            "Conditioned" if model.Basement.Conditioned else "Unconditioned"
        )
        dumped_self["feature.extra_spaces.attic.occupied"] = (
            model.Attic.UseFraction or 0
        )
        dumped_self["feature.extra_spaces.attic.conditioned"] = (
            "Conditioned" if model.Attic.Conditioned else "Unconditioned"
        )

        index = pd.MultiIndex.from_tuples(
            [tuple(dumped_self.values())],
            names=list(dumped_self.keys()),
        )
        results = results.to_frame().T.set_index(index)
        return idf, results, err_text


if __name__ == "__main__":
    import time

    from epinterface.data import DefaultEPWZipPath

    artifact_path = Path("E:/repos/epinterface/tests/data").as_posix()
    spec = SBEMSimulationSpec(
        experiment_id="test-2",
        sort_index=0,
        rotated_rectangle="POLYGON ((5 0, 5 10, 15 10, 15 0, 5 0))",
        num_floors=3,
        height=3 * 3.5,
        long_edge=17,
        short_edge=17,
        aspect_ratio=1,
        rotated_rectangle_area_ratio=1,
        long_edge_angle=0.23,
        wwr=0.15,
        f2f_height=3.5,
        neighbor_polys=["POLYGON ((-10 0, -10 10, -5 10, -5 0, -10 0))"],
        neighbor_floors=[3],
        neighbor_heights=[10.5],
        epwzip_uri=f"file://{DefaultEPWZipPath}",  # pyright: ignore [reportArgumentType]
        db_uri=AnyUrl(f"file://{artifact_path}/components-ma-with-abs.db"),
        semantic_fields_uri=AnyUrl(
            f"file://{artifact_path}/semantic-fields-ma-with-abs.yml"
        ),
        component_map_uri=AnyUrl(
            f"file://{artifact_path}/component-map-ma-with-abs.yml"
        ),
        semantic_field_context={
            "Region": "MA",
            "Typology": "SFH",
            "Age_bracket": "post_2003",
            "AtticVentilation": "UnventilatedAttic",
            "AtticFloorInsulation": "UninsulatedFloor",
            "RoofInsulation": "LimitedInsulationRoof",
            "BasementCeilingInsulation": "UninsulatedCeiling",
            "BasementWallsInsulation": "UninsulatedWalls",
            "GroundSlabInsulation": "UninsulatedGroundSlab",
            "Weatherization": "TightEnvelope",
            "Walls": "FullInsulationWallsCavity",
            "Windows": "DoublePaneLowE",
            "Heating": "NaturalGasHeating",
            "Cooling": "ACWindow",
            "Distribution": "HotWaterUninsulated",
            "DHW": "NaturalGasDHW",
            "Lighting": "LED",
            "Thermostat": "Controls",
            "Equipment": "HighEfficiencyEquipment",
        },
    )

    s = time.time()
    idf, results, warnings = spec.run()
    e = time.time()
    print(f"Execution time: {e - s:.2f} seconds")
    print("----")
    print(
        results.reset_index(drop=True)["Raw"]
        .stack(level="Month", future_stack=True)
        .sum()
    )
    print(results["Raw"].sum().sum())
    print("----")
    print("----")
    print(
        results.reset_index(drop=True)["End Uses"]
        .stack(level="Month", future_stack=True)
        .sum()
    )
    print(results["End Uses"].sum().sum())
    print("----")
    print("----")
    print(
        results.reset_index(drop=True)["Utilities"]
        .stack(level="Month", future_stack=True)
        .sum()
    )
    print(results["Utilities"].sum().sum())
    print("----")
    idf.saveas(f"{artifact_path}/test.idf")
    # print(results.reset_index(drop=True)["End Uses"])
    # print("----")
    # print(results.reset_index(drop=True)["Utilities"])
    # print("----")
