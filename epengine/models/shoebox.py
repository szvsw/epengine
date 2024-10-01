"""Shoebox simulation workflow."""

import json
from pathlib import Path

import pandas as pd
from archetypal.idfclass import IDF
from epinterface.climate_studio.builder import Model
from epinterface.climate_studio.interface import ClimateStudioLibraryV2
from epinterface.geometry import ShoeboxGeometry, match_idf_to_building_and_neighbors
from epinterface.weather import WeatherUrl
from pydantic import AnyUrl, Field

from epengine.models.base import LeafSpec


class ShoeboxSimulationSpec(LeafSpec):
    """A spec for running a shoebox simulation."""

    lib_uri: AnyUrl = Field(..., description="The uri of the library file to fetch.")
    typology: str = Field(..., description="The typology of the building to simulate.")
    year_built: int = Field(..., description="The year the building was built.")
    num_floors: int = Field(..., description="The number of floors in the building.")
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
    rotated_rectangle_area_ratio: float = Field(
        ..., description="The area ratio of the rotated rectangle."
    )

    @property
    def lib_path(self) -> Path:
        """Fetch the library file and return the local path.

        Returns:
            local_path (Path): The local path of the fetched library file
        """
        return self.fetch_uri(self.lib_uri)

    @property
    def lib(self) -> ClimateStudioLibraryV2:
        """Fetch the library file and return the library.

        Returns:
            lib (ClimateStudioLibraryV2): The fetched library
        """
        with open(self.lib_path) as f:
            lib_data = json.load(f)
            return ClimateStudioLibraryV2.model_validate(lib_data)

    def configure(self, f2f_height: float = 3.5) -> Model:
        """Configure the model and return it.

        Args:
            f2f_height (float): The floor-to-floor height of the building

        Returns:
            model (epinterface.climate_studio.builder.Model): The configured model
        """
        # TODO: decide whether we should actually go up to core/perim or not
        use_core_perim = self.long_edge > 15 and self.short_edge > 15
        geometry = ShoeboxGeometry(
            x=0,
            y=0,
            w=10,
            d=10,
            h=f2f_height,
            num_stories=self.num_floors,
            zoning="core/perim" if use_core_perim else "by_storey",
            perim_depth=3,
            roof_height=None,
            basement_depth=None,
            wwr=0.15,
        )
        if AnyUrl(self.epwzip_path).scheme == "D":
            parts = list(Path(self.epwzip_path).parts)
            pth = "/".join(parts[2:])
            self.epwzip_path = f"https://climate.onebuilding.org/{pth}"

        # TODO: select space use and envelope based off of
        # typology age and size
        model = Model(
            Weather=WeatherUrl(self.epwzip_path),  # pyright: ignore [reportCallIssue]
            geometry=geometry,
            space_use_name=next(iter(self.lib.SpaceUses)),
            envelope_name=next(iter(self.lib.Envelopes)),
            lib=self.lib,
        )
        return model

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
        experiment_id="test",
        sort_index=0,
        lib_uri=AnyUrl("s3://ml-for-bem/tiles/massachusetts/2024_09_30/lib_demo.json"),
        typology="Residential",
        year_built=1972,
        num_floors=3,
        neighbor_polys=["POLYGON ((-10 0, -10 10, -5 10, -5 0, -10 0))"],
        neighbor_floors=[3],
        rotated_rectangle="POLYGON ((5 0, 5 10, 15 10, 15 0, 5 0))",
        rotated_rectangle_area_ratio=1,
        long_edge=10,
        short_edge=10,
        long_edge_angle=0.23,
        footprint_area=100,
        epwzip_path="https://climate.onebuilding.org/WMO_Region_4_North_and_Central_America/USA_United_States_of_America/MA_Massachusetts/USA_MA_Boston-Logan.Intl.AP.725090_TMYx.2009-2023.zip",
    )

    idf, results, warnings = spec.run()
    print(results)
