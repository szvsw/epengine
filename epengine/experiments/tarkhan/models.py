"""Nada Tarkhan's UWG simulation experiments."""

import json
import os
import re
import shutil
import subprocess
import tempfile
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
from eppy.bunch_subclass import BadEPFieldError
from eppy.modeleditor import IDF
from pydantic import AnyUrl, Field

from epengine.models.base import LeafSpec
from epengine.utils.results import make_onerow_multiindex_from_dict

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client
else:
    S3Client = Any


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

    def run(self, s3: S3Client):
        """Run the simulation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            # idf_path = temp_dir / idf_path.name
            # epw_path = temp_dir / epw_path.name
            safe_idf_path = temp_dir / self.idf_path.name
            safe_epw_path = temp_dir / self.epw_path.name
            shutil.copy(self.idf_path, safe_idf_path)
            shutil.copy(self.epw_path, safe_epw_path)
            # TODO: add zone_metadata to the results df
            summary, _zone_metadata, hourly_results = self.do_work_in_temp_dir(
                safe_idf_path, safe_epw_path, temp_dir
            )

        summary = pd.DataFrame([summary], index=self.make_multiindex())
        hourly_results = hourly_results.T
        hourly_results.index.name = "Zone"
        hourly_results = hourly_results.set_index(
            self.make_multiindex(n_rows=len(hourly_results)), append=True
        )

        results_key = f"{self.scoped_prefix}/results/{self.building_name}/results.h5"
        bucket = str(self.idf_uri).split("/")[2]
        results_uri = f"s3://{bucket}/{results_key}"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            results_path = temp_dir / "results.h5"
            summary.to_hdf(results_path, key="summary", mode="w")
            hourly_results.to_hdf(results_path, key="hourly_results", mode="a")
            s3.upload_file(results_path.as_posix(), bucket, results_key)

        # TODO: alternatively, return a normal dict with just the summary
        # but add a reference to the hourly_results_uri in the summary ix
        return {"uri": results_uri}

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
        # Path to EnergyPlus and IDD
        ENERGYPLUS_EXE = "/usr/local/EnergyPlus-24-2-0/energyplus"
        IDD_PATH = "/usr/local/EnergyPlus-24-2-0/Energy+.idd"
        if not Path(ENERGYPLUS_EXE).exists():
            msg = f"EnergyPlus executable {ENERGYPLUS_EXE} does not exist"
            raise ValueError(msg)
        if not Path(IDD_PATH).exists():
            msg = f"IDD file {IDD_PATH} does not exist"
            raise ValueError(msg)

        # Ensure Eppy sees the correct IDD
        IDF.setiddname(IDD_PATH)

        # load the IDF
        idf = IDF(idf_path.as_posix())

        # Mutate the IDF
        update_idf_version(idf, "24.2")
        add_temp_humidity_outputs(idf)

        # extract metadata from the IDF
        # TODO: replace this with actual metadata method
        zone_metadata: pd.DataFrame = self.create_zone_data(idf)

        # Save the IDF (maybe)
        mutated_idf_path = temp_dir / "mutated.idf"
        results_dir = temp_dir / "results"
        os.makedirs(results_dir, exist_ok=True)
        idf.saveas(mutated_idf_path.as_posix())

        # Run the simulation
        success, proc = run_energyplus(
            ENERGYPLUS_EXE,
            mutated_idf_path.as_posix(),
            epw_path.as_posix(),
            results_dir.as_posix(),
        )
        if not success:
            msg = f"EnergyPlus simulation failed for {idf_path}\n"
            msg += f"stdout: {proc.stdout}\n"
            msg += f"stderr: {proc.stderr}\n"
            raise ValueError(msg)

        # Get the results
        # TODO: replace this with actual path and any other required arguments to extract_hourly_results
        results_file_path = results_dir / "eplusout.csv"
        hourly_results = extract_hourly_csv(results_file_path)

        # convert the results to a dictionary of dataframes
        summary = compute_summary_stats(hourly_results)

        return (summary, zone_metadata, hourly_results)

    def make_multiindex(
        self,
        additional_index_data: dict[str, Any] | None = None,
        n_rows: int = 1,
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

        try:
            json.dumps(index_data)
        except Exception as e:
            msg = f"Index data is not JSON serializable: {e}"
            raise ValueError(msg) from e

        return make_onerow_multiindex_from_dict(index_data, n_rows)

    def create_zone_data(self, idf: IDF) -> pd.DataFrame:  # noqa: C901
        """Generates a CSV of zone-level data (orientation + floor assignment) for the given IDF."""
        zone_azimuths = {}
        zones_seen = set()

        # Collect orientation data
        for surface in idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
            zname = normalize_zone_name(surface.Zone_Name)
            zones_seen.add(zname)
            if (
                surface.Surface_Type
                and surface.Surface_Type.lower() == "wall"
                and surface.Outside_Boundary_Condition
                and surface.Outside_Boundary_Condition.lower() == "outdoors"
            ):
                try:
                    az = float(surface.azimuth) if surface.azimuth else 0.0
                except (ValueError, BadEPFieldError):
                    az = 0.0
                zone_azimuths.setdefault(zname, []).append(az)

        # Determine floor assignment
        def extract_floor_number(z):
            pattern = re.compile(r"(?:Floor|Flr)(\d+)", re.IGNORECASE)
            match = pattern.search(z)
            return int(match.group(1)) if match else 0

        zone_floor_nums = {}
        for z in zones_seen:
            zone_floor_nums[z] = extract_floor_number(z)

        if zone_floor_nums:
            min_floor = min(zone_floor_nums.values())
            max_floor = max(zone_floor_nums.values())
        else:
            min_floor = max_floor = 0

        def get_floor_assignment(num, mn, mx):
            if mn == mx or num == mn:
                return "ground"
            elif num == mx:
                return "top"
            else:
                return "middle"

        # Build records
        records = []
        for z in zones_seen:
            az_list = zone_azimuths.get(z, [])
            if az_list:
                avg_az = sum(az_list) / len(az_list)
                orient = get_orientation(avg_az)
            else:
                avg_az = 0.0
                orient = "core"

            floor_num = zone_floor_nums[z]
            floor_asst = get_floor_assignment(floor_num, min_floor, max_floor)

            records.append({
                "normalized_zone_name": z,
                "avg_azimuth": round(avg_az, 1),
                "orientation": orient,
                "floor_number": floor_num,
                "floor_assignment": floor_asst,
            })

        df = pd.DataFrame(records)
        return df

    @property
    def scoped_prefix(self) -> str:
        """The scoped prefix for the experiment."""
        bucket_prefix = "hatchet"
        return format_scoped_prefix(
            bucket_prefix, self.experiment_id, self.case, self.city, self.grid
        )


def format_scoped_prefix(
    bucket_prefix: str, experiment_id: str, case_id: str, city: str, grid_id: str
) -> str:
    """Format a scoped prefix for a bucket."""
    return f"{bucket_prefix}/{experiment_id}/{case_id}/{city}/{grid_id}"


def update_idf_version(idf: IDF, desired_version="24.2"):
    """Updates (or creates) the VERSION object in the IDF."""
    if "VERSION" in idf.idfobjects:
        version_obj = idf.idfobjects["VERSION"][0]
        version_obj.Version_Identifier = desired_version
    else:
        idf.newidfobject("VERSION", Version_Identifier=desired_version)


def add_temp_humidity_outputs(idf: IDF):
    """Ensures 'Zone Air Temperature' and 'Zone Air Relative Humidity' are reported hourly."""
    required_variables = ["Zone Air Temperature", "Zone Air Relative Humidity"]
    for var in required_variables:
        found = any(
            obj.Variable_Name == var for obj in idf.idfobjects["OUTPUT:VARIABLE"]
        )
        if not found:
            idf.newidfobject(
                "OUTPUT:VARIABLE",
                Key_Value="*",
                Variable_Name=var,
                Reporting_Frequency="Hourly",
            )


def get_orientation(azimuth_degrees):
    """Get the orientation of the building based on the azimuth."""
    az = azimuth_degrees % 360
    if az >= 315 or az < 45:
        return "North"
    elif 45 <= az < 135:
        return "East"
    elif 135 <= az < 225:
        return "South"
    elif 225 <= az < 315:
        return "West"
    return "Unknown"


def normalize_zone_name(zname):
    """Normalize the zone name to a standard format."""
    known_prefixes = ["Flr1_", "Flr2_", "Flr3_", "Flr4_"]
    for prefix in known_prefixes:
        if zname.startswith(prefix):
            return zname.replace(prefix, "")
    return zname


def run_energyplus(EPLUS_EXE: str, idf_path: str, epw_path: str, output_dir: str):
    """Runs E+ and saves outputs to output_dir. Returns True if successful."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [EPLUS_EXE, "-w", epw_path, "-d", output_dir, "-r", idf_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
    return proc.returncode == 0, proc


def compute_summary_stats(hourly_df: pd.DataFrame) -> pd.Series:
    """Reads the hourly T & RH df, calculate summary stats.

    In particular, we calculate:
      - peak_temp (max T across all zones/hours)
      - coldest_temp (min T)
      - overheat_hours (# hours with max T > 26)
      - cold_hours (# hours with min T < 10)

    Args:
        hourly_df (pd.DataFrame): The hourly T & RH df.

    Returns:
        pd.Series: A series of the summary stats.
    """
    temp_cols = [c for c in hourly_df.columns if "temperature" in c.lower()]
    if not temp_cols:
        return pd.Series({
            "peak_temp": None,
            "coldest_temp": None,
            "overheat_hours": None,
            "cold_hours": None,
        })

    hourly_df["max_T"] = hourly_df[temp_cols].max(axis=1)
    hourly_df["min_T"] = hourly_df[temp_cols].min(axis=1)

    peak_temp = hourly_df["max_T"].max()  # highest zone T
    coldest_temp = hourly_df["min_T"].min()  # lowest zone T
    overheat_hours = (hourly_df["max_T"] > 26).sum()
    cold_hours = (hourly_df["min_T"] < 10).sum()

    return pd.Series({
        "peak_temp": round(peak_temp, 2),
        "coldest_temp": round(coldest_temp, 2),
        "overheat_hours": int(overheat_hours),
        "cold_hours": int(cold_hours),
    })


def extract_hourly_csv(hourly_csv_path: Path) -> pd.DataFrame:
    """Parses eplusout.csv for T & RH columns.

    Args:
        hourly_csv_path (Path): The path to the eplusout.csv file.

    Returns:
        pd.DataFrame: A dataframe of the hourly T & RH data.
    """
    if not hourly_csv_path.exists():
        msg = f"eplusout.csv not found in {hourly_csv_path}"
        raise ValueError(msg)

    df = pd.read_csv(hourly_csv_path)
    # Identify temperature / RH columns
    temp_cols = [c for c in df.columns if "temperature" in c.lower()]
    rh_cols = [c for c in df.columns if "relative humidity" in c.lower()]
    if not temp_cols and not rh_cols:
        msg = f"No Temp/RH columns found in {hourly_csv_path}."
        raise ValueError(msg)

    time_cols = [
        c for c in df.columns if "date/time" in c.lower() or "datetime" in c.lower()
    ]
    selected_cols = time_cols + temp_cols + rh_cols
    df_out = df[selected_cols].copy()
    return cast(pd.DataFrame, df_out)
