"""Nada Tarkhan's UWG simulation experiments."""

import json
import re
import shutil
import tempfile
import zipfile
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import AnyUrl, Field

from epengine.models.base import LeafSpec
from epengine.utils.results import make_onerow_multiindex_from_dict


class TarkhanSpec(LeafSpec):
    """A spec for running an EnergyPlus simulation."""

    idf_uri: AnyUrl = Field(
        ..., description="The uri of the idf file to fetch and simulate"
    )
    epw_uri: AnyUrl = Field(
        ..., description="The uri of the epw file to use during simulation"
    )
    city: str = Field(..., description="The city of the building")
    case: str = Field(..., description="The id of the simulation")
    grid: str = Field(..., description="The grid of the building")
    climate: str = Field(..., description="The climate of the building")
    building_name: str = Field(..., description="The name of the building")

    @cached_property
    def idf_path(self):
        """The path to the scoped IDF path."""
        return self.fetch_uri(self.idf_uri)

    @cached_property
    def epw_path(self):
        """The path to the scoped EPW path."""
        return self.fetch_uri(self.epw_uri)

    def run(self):
        """Run the simulation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            # idf_path = temp_dir / idf_path.name
            # epw_path = temp_dir / epw_path.name
            safe_idf_path = temp_dir / self.idf_path.name
            safe_epw_path = temp_dir / self.epw_path.name
            shutil.copy(self.idf_path, safe_idf_path)
            shutil.copy(self.epw_path, safe_epw_path)
            results = self.do_work_in_temp_dir(safe_idf_path, safe_epw_path, temp_dir)

        # results = serialize_df_dict(results)
        return results

    def do_work_in_temp_dir(
        self, idf_path: Path, epw_path: Path, temp_dir: Path
    ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        """Do all simulation in a safe temporary directory.

        Args:
            idf_path (Path): The resolved path to the idf file (e.g. /tmp/12837/Residential_17.idf)
            epw_path (Path): The resolved path to the epw file (e.g. /tmp/12837/C1_CHICAGO_G2_UWG_TMY.epw)
            temp_dir (Path): The temporary directory which can be safely written to if needed (e.g. /tmp/12837)

        Returns:
            results_dataframes (dict[str, pd.DataFrame]): A dictionary of dataframes, keyed by the name of the output variable
        """
        # load the IDF

        # Mutate the IDF

        # extract metadata from the IDF
        # TODO: replace this with actual metadata method
        zone_metadata: pd.DataFrame = pd.DataFrame()

        # Save the IDF (maybe)

        # Run the simulation

        # Get the results
        # TODO: replace this with actual path and any other required arguments to extract_hourly_results
        results_file_path = temp_dir / "path" / "to" / "eplusout.csv"
        hourly_results: pd.DataFrame = self.extract_hourly_results(results_file_path)

        # convert the results to a dictionary of dataframes

        summary = pd.Series({
            "peak_temp": 10,
            "overheat_hours": 100,
            "coldest_temp": 100,
            "cold_hours": 100,
        })

        return (summary, zone_metadata, hourly_results)

    def make_multiindex(
        self,
        additional_index_data: dict[str, Any] | None = None,
    ) -> pd.MultiIndex:
        """Make a MultiIndex from the Spec, and any other methods which might create index data.

        Note that index data should generally be considered as features or inputs, rather than outputs.

        TODO: Feel free to add more args to this method if more values need to be computed.

        Returns:
            multi_index (pd.MultiIndex): The MultiIndex.
        """
        index_data: dict[str, Any] = self.model_dump(mode="json")
        if additional_index_data:
            index_data.update(additional_index_data)

        # TODO: add more data if desired, e.g.
        # index_data["building_age"] = "pre_1975"
        # or
        # index_data.update(self.make_some_other_dict())

        try:
            json.dumps(index_data)
        except Exception as e:
            msg = f"Index data is not JSON serializable: {e}"
            raise ValueError(msg) from e

        return make_onerow_multiindex_from_dict(index_data)

    def extract_hourly_results(
        self,
        results_file_path: Path,
    ) -> pd.DataFrame:
        """Extract the hourly results from the simulation.

        Nb: you may wish to add additional arguments to this method.  Feel free to do so!

        Pass in the results file path that you want to extract data from.

        Args:
            results_file_path (Path): The path to the results file to extract data from.

        Returns:
            hourly_results (pd.DataFrame): The hourly results from the simulation.
        """
        return pd.DataFrame()


def make_experiment_specs(
    cases_zipfolder: Path,
    experiment_id: str,
    bucket: str = "ml-for-bem",
    bucket_prefix: str = "hatchet",
):
    """Make experiment specs from a zip file of cases."""
    # make sure cases_zipfolder is a zip file
    if cases_zipfolder.suffix != ".zip":
        msg = f"Cases zip folder {cases_zipfolder} is not a zip file"
        raise ValueError(msg)

    with tempfile.TemporaryDirectory() as temp_dir:
        tempdir = Path(temp_dir)
        cases_folder = tempdir / cases_zipfolder.stem
        # unzip the cases_zipfolder
        with zipfile.ZipFile(cases_zipfolder, "r") as zip_ref:
            zip_ref.extractall(cases_folder)

        # create a reg ex to confirm that it matches the pattern C[some number of digits]_[some number of any characters]_G[some number of digits]
        regex = re.compile(r"^C\d+_.*_G\d+$")
        all_subfolders = list(filter(lambda p: p.is_dir(), cases_folder.iterdir()))
        all_subfolder_names = [p.name for p in all_subfolders]
        matching_subfolder_names = list(filter(regex.match, all_subfolder_names))
        matching_subfolders = [
            p for p in all_subfolders if p.name in matching_subfolder_names
        ]
        # create a regex to grab the C value from the start of the string and the G value from the end of the string
        regex = re.compile(r"^C(\d+)_(.*)_G(\d+)$")
        all_specs: list[TarkhanSpec] = []
        for subfolder in matching_subfolders:
            match = regex.match(subfolder.name)
            if match:
                case_id = match.group(1)
                city = match.group(2)
                grid_id = match.group(3)
                epws = list(subfolder.glob("*.epw"))
                if len(epws) != 1:
                    msg = f"Expected 1 epw file in {subfolder}, found {len(epws)}"
                    raise ValueError(msg)
                epw = epws[0]
                climate_regex = re.compile(r"^C\d+_.*_G\d+_(.*).epw$")
                climate_matches = climate_regex.match(epw.name)
                if not climate_matches:
                    msg = f"Expected epw file to match regex {climate_regex}, but it did not"
                    raise ValueError(msg)
                climate = climate_matches.group(1)
                idfs = list(subfolder.glob("*.idf"))
                for idf in idfs:
                    idf_name = idf.stem
                    idf_key = f"{bucket_prefix}/{experiment_id}/{case_id}/{city}/{grid_id}/{idf_name}.idf"
                    idf_uri = f"s3://{bucket}/{idf_key}"
                    epw_key = f"{bucket_prefix}/{experiment_id}/{case_id}/{city}/{grid_id}/{epw.name}"
                    epw_uri = f"s3://{bucket}/{epw_key}"
                    spec = TarkhanSpec(
                        sort_index=0,
                        experiment_id=experiment_id,
                        idf_uri=AnyUrl(idf_uri),
                        epw_uri=AnyUrl(epw_uri),
                        case=case_id,
                        grid=grid_id,
                        climate=climate,
                        building_name=idf_name,
                        city=city,
                    )
                    all_specs.append(spec)
        df = pd.DataFrame([s.model_dump() for s in all_specs])
        return df


if __name__ == "__main__":
    cases_zipfolder = (
        Path(__file__).parent.parent.parent.parent
        / "local_artifacts"
        / "experiments"
        / "tarkhan"
        / "Cases.zip"
    )
    res = make_experiment_specs(
        cases_zipfolder,
        experiment_id="tarkhan/test",
    )
    print(res)
