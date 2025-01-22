"""Shoebox simulation workflow."""

import json
import math
from functools import cached_property
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from archetypal.idfclass import IDF
from epinterface.actions import ActionLibrary, ActionSequence
from epinterface.climate_studio.builder import Model
from epinterface.climate_studio.interface import ClimateStudioLibraryV2
from epinterface.geometry import ShoeboxGeometry, match_idf_to_building_and_neighbors
from epinterface.weather import WeatherUrl
from pydantic import AnyUrl, Field, field_serializer, model_validator

from epengine.models.base import LeafSpec


class ShoeboxSimulationSpec(LeafSpec):
    """A spec for running a shoebox simulation."""

    lib_uri: AnyUrl = Field(..., description="The uri of the library file to fetch.")
    retrofit_lib_uri: AnyUrl | None = Field(
        default=None, description="The uri of the library file to upgrade."
    )
    retrofit: str | None = Field(
        default=None,
        description="The selection of the retrofit to apply.  Must be present in the provided lib.",
    )
    typology: str = Field(..., description="The typology of the building to simulate.")
    year_built: int = Field(..., description="The year the building was built.")
    num_floors: float = Field(..., description="The number of floors in the building.")
    rotated_rectangle: str = Field(
        ..., description="The rotated rectangle of the building."
    )
    neighbor_polys: list[str] = Field(
        ..., description="The polygons of the neighboring buildings."
    )
    neighbor_floors: list[float] = Field(
        ..., description="The number of floors of the neighboring buildings."
    )
    epwzip_path: str = Field(..., description="The path to the epwzip file.")
    footprint_area: float = Field(
        ..., description="The footprint area of the building."
    )
    long_edge_angle: float = Field(
        ..., description="The long edge angle of the building (radians)."
    )
    long_edge: float = Field(
        ..., description="The length of the long edge of the building."
    )
    short_edge: float = Field(
        ..., description="The length of the short edge of the building."
    )

    @field_serializer("retrofit")
    def convert_None_to_baseline_string(self, v):
        """Convert None to an empty string for the retrofit field."""
        return v if v is not None else "Baseline"

    @model_validator(mode="after")
    def check_retrofit_uri_is_not_none_when_retrofit_is_not_none(self):
        """Check that retrofit_uri is not None when retrofit is not None."""
        if self.retrofit and not self.retrofit_lib_uri:
            raise ValueError("RETROFIT_URI_REQUIRED")
        return self

    @cached_property
    def lib_path(self) -> Path:
        """Fetch the library file and return the local path.

        Returns:
            local_path (Path): The local path of the fetched library file
        """
        return self.fetch_uri(self.lib_uri)

    @cached_property
    def retrofit_lib_path(self) -> Path | None:
        """Fetch the retrofit library file and return the local path.

        Returns:
            local_path (Path): The local path of the fetched retrofit library file
        """
        return self.fetch_uri(self.retrofit_lib_uri) if self.retrofit_lib_uri else None

    @cached_property
    def lib(self) -> ClimateStudioLibraryV2:
        """Fetch the library file and return the library.

        Returns:
            lib (ClimateStudioLibraryV2): The fetched library
        """
        with open(self.lib_path) as f:
            lib_data = json.load(f)
            return ClimateStudioLibraryV2.model_validate(lib_data)

    @cached_property
    def retrofit_lib(self) -> ActionLibrary | None:
        """Fetch the retrofit library file and return the library.

        Returns:
            lib (ClimateStudioLibraryV2): The fetched retrofit library
        """
        if self.retrofit_lib_path:
            with open(self.retrofit_lib_path) as f:
                lib_data = yaml.safe_load(f)
                return ActionLibrary.model_validate(lib_data)
        return None

    def select_retrofit(self) -> ActionSequence | None:
        """Select the retrofit from the retrofit library.

        Returns:
            retrofit (ActionSequence | None): The selected retrofit
        """
        if (
            self.retrofit is None
            or self.retrofit_lib is None
            or self.retrofit.lower()
            == "baseline"  # TODO: remove once NaN/None serialization is fixed.
        ):
            return None
        retrofit_suffix = (
            f" {self.size_key}" if self.retrofit.lower().startswith("deep") else ""
        )
        retrofit_name = f"{self.retrofit}{retrofit_suffix}"
        retrofit = self.retrofit_lib.get(retrofit_name)
        return retrofit

    def configure(self, f2f_height: float = 3.5) -> Model:
        """Configure the model and return it.

        Args:
            f2f_height (float): The floor-to-floor height of the building

        Returns:
            model (epinterface.climate_studio.builder.Model): The configured model
        """
        if not self.is_residential:
            raise ValueError("RESIDENTIAL_ONLY")

        # TODO: decide whether we should actually go up to core/perim or not
        use_core_perim = self.long_edge > 15 and self.short_edge > 15
        has_basement = True
        should_condition_basement = False
        wwr = 0.15
        geometry = ShoeboxGeometry(
            x=0,
            y=0,
            w=self.short_edge,
            d=self.long_edge,
            h=f2f_height,
            num_stories=math.ceil(self.num_floors),
            zoning="core/perim" if use_core_perim else "by_storey",
            perim_depth=3,
            roof_height=None,
            basement=has_basement,
            wwr=wwr,
        )

        self.update_windows_by_age(self.envelope_name)

        model = Model(
            Weather=WeatherUrl(self.epwzip_path),  # pyright: ignore [reportCallIssue]
            geometry=geometry,
            space_use_name=self.space_use_name,
            envelope_name=self.envelope_name,
            conditioned_basement=should_condition_basement,
            lib=self.lib,
        )
        if retrofit := self.select_retrofit():
            retrofit.run(model)
        return model

    @property
    def is_residential(self) -> bool:
        """Check if the typology is residential.

        Returns:
            is_residential (bool): True if the typology is residential
        """
        return "residential" in self.typology.lower()

    @property
    def res_size(self) -> Literal["multi-family", "single-family"]:
        """Return the size of the residential building.

        Returns:
            res_size (Literal["multi-family", "single-family"]): The size of the building
        """
        # TODO: decide on the threshold values
        if (
            (
                ("multi" in self.typology.lower() or "mf" in self.typology.lower())
                and "4 units" not in self.typology.lower()
                and "3 units" not in self.typology.lower()
            )
            or self.num_floors > 3
            or self.footprint_area > 1000
        ):
            return "multi-family"
        return "single-family"

    @property
    def age_key(self):
        """Return the age key of the building.

        Returns:
            age_key (str): The age key of the building
        """
        return (
            "pre_1975"
            if self.year_built < 1975
            else ("btw_1975_2003" if self.year_built <= 2003 else "post_2003")
        )

    @property
    def size_key(self):
        """Return the size key of the building.

        Returns:
            size_key (str): The size key of the building
        """
        return "MF" if self.res_size == "multi-family" else "SF"

    @property
    def space_use_name(self):
        """Return the space use name of the building.

        Returns:
            space_use_name (str): The space use name of the building
        """
        space_use_name = f"MA_{self.size_key}_{self.age_key}"
        return space_use_name

    @property
    def envelope_name(self):
        """Return the envelope name of the building.

        Returns:
            envelope_name (str): The envelope name of the building
        """
        envelope_name = f"MA_{self.size_key}_{self.age_key}"
        return envelope_name

    def update_windows_by_age(self, envelope_name: str):
        """Update the windows based off of the age of the building.

        NB: this mutates the library in place.

        Args:
            envelope_name (str): The envelope name to update
        """
        if self.year_built < 1975:
            window_name = "Template_pre_1975"
        elif self.year_built <= 2003:
            window_name = "Template_btw_1975_2003"
        else:
            window_name = "Template_post_2003"

        # select window type based off of year
        envelope = self.lib.Envelopes[envelope_name]
        if envelope.WindowDefinition is None:
            raise ValueError("Envelope:WindowDefinition:MISSING")
        if window_name not in self.lib.GlazingConstructions:
            raise ValueError(f"GlazingConstruction:NOT_FOUND:{window_name}")

        envelope.WindowDefinition.Construction = window_name

    def run(self):
        """Build and run the shoebox simulation."""
        f2f_height = 3.5
        model = self.configure(f2f_height=f2f_height)

        def post_build_callback(idf: IDF) -> IDF:
            idf = match_idf_to_building_and_neighbors(
                idf,
                building=self.rotated_rectangle,
                neighbor_polys=self.neighbor_polys,  # pyright: ignore [reportArgumentType]
                neighbor_floors=self.neighbor_floors,  # pyright: ignore [reportArgumentType]
                neighbor_f2f_height=f2f_height,
                target_long_length=self.long_edge,
                target_short_length=self.short_edge,
                rotation_angle=self.long_edge_angle,
            )
            return idf

        # weather_dir = self.local_path(AnyUrl(self.epwzip_path)).parent
        weather_dir = Path("notebooks") / "cache" / "weather"
        weather_dir.mkdir(parents=True, exist_ok=True)
        idf, results, warning_text = model.run(
            weather_dir=weather_dir,
            move_energy=False,
            post_build_callback=post_build_callback,
        )
        dumped_self = self.model_dump(
            exclude={"neighbor_polys", "neighbor_floors"}, mode="json"
        )
        index = pd.MultiIndex.from_tuples(
            [tuple(dumped_self.values())],
            names=list(dumped_self.keys()),
        )
        results = results.to_frame().T.set_index(index)
        return idf, results, warning_text


if __name__ == "__main__":
    spec = ShoeboxSimulationSpec(
        experiment_id="test-2",
        sort_index=0,
        lib_uri=AnyUrl(
            "s3://ml-for-bem/tiles/massachusetts/2024_09_30/everett_lib.json"
        ),
        retrofit=None,
        retrofit_lib_uri=AnyUrl(
            "s3://ml-for-bem/tiles/massachusetts/2024_09_30/everett_retrofits.yaml"
        ),
        typology="Residential",
        year_built=1950,
        num_floors=2,
        neighbor_polys=["POLYGON ((-10 0, -10 10, -5 10, -5 0, -10 0))"],
        neighbor_floors=[3],
        rotated_rectangle="POLYGON ((5 0, 5 10, 15 10, 15 0, 5 0))",
        long_edge=10,
        short_edge=10,
        long_edge_angle=0.23,
        footprint_area=100,
        epwzip_path="https://climate.onebuilding.org/WMO_Region_4_North_and_Central_America/USA_United_States_of_America/MA_Massachusetts/USA_MA_Boston-Logan.Intl.AP.725090_TMYx.2009-2023.zip",
    )

    idf, results, warnings = spec.run()
    print(results.reset_index(drop=True))
