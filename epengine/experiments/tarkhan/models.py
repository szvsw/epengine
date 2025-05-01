"""Nada Tarkhan's UWG simulation experiments."""

import json
import os
import re
import shutil
import subprocess
import tempfile
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from eppy.bunch_subclass import BadEPFieldError
from eppy.modeleditor import IDF
from pydantic import AnyUrl, Field

from epengine.models.base import LeafSpec
from epengine.utils.results import make_onerow_multiindex_from_dict, serialize_df_dict

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
            summary, zone_metadata, hourly_results = self.do_work_in_temp_dir(
                safe_idf_path, safe_epw_path, temp_dir
            )

            # the summary only has a single row, so we can just use the multiindex
            summary = pd.DataFrame([summary], index=self.make_multiindex())

            # we will move the normalized zone name into the index, and then append the
            # multiindex with the appropriate number of rows
            zone_metadata = zone_metadata.set_index("normalized_zone_name").set_index(
                self.make_multiindex(n_rows=len(zone_metadata)), append=True
            )

            # the hourly result needs to be transformed and we add a title to the index,
            # which is currently the zone name.  Then we will drop the redundant
            # date/time row since the columns are already in the correct order.
            # finally we add the multiindex with the appropriate number of rows
            # to store the grid/case/city/building config.
            hourly_results = hourly_results.T
            hourly_results.index.name = "Zone"
            hourly_results = hourly_results.rename(columns=lambda x: str(x))
            hourly_results = hourly_results.set_index(
                self.make_multiindex(n_rows=len(hourly_results)), append=True
            )

        # serialize the hourly results to s3 as a parquet file
        bucket = str(self.idf_uri).split("/")[2]
        hourly_results_key = (
            f"{self.scoped_prefix}/results/{self.building_name}/hourly_results.pq"
        )
        hourly_results_uri = f"s3://{bucket}/{hourly_results_key}"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            hourly_results_path = temp_dir / "hourly_results.pq"
            hourly_results.to_parquet(hourly_results_path)
            s3.upload_file(hourly_results_path.as_posix(), bucket, hourly_results_key)
        # hourly_results.to_parquet(hourly_results_uri)

        # inject the hourly results uri into the summary index so i t can be easily used
        # in the future to fetch the hourly results
        summary = summary.set_index(
            pd.Series(
                [hourly_results_uri],
                index=["hourly_results_uri"],
                name="hourly_results_uri",
            ),
            append=True,
        )

        # _results_key = f"{self.scoped_prefix}/results/{self.building_name}/results.h5"
        # _results_uri = f"s3://{bucket}/{_results_key}"
        # with tempfile.TemporaryDirectory() as temp_dir:
        #     temp_dir = Path(temp_dir)
        #     results_path = temp_dir / "results.h5"
        #     summary.to_hdf(results_path, key="summary", mode="w")
        #     s3.upload_file(results_path.as_posix(), bucket, results_key)
        #     hourly_results_path = temp_dir / "hourly_results.pq"
        #     s3.upload_file(hourly_results_path.as_posix(), bucket, hourly_results_key)

        results = {"summary": summary, "zone_metadata": zone_metadata}
        return serialize_df_dict(results)

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
        idf, els_for_azimuth = simplify_shading(idf)
        # building_coords: list[np.ndarray] = []
        # for key in idf.idfobjects:
        #     if "buildingsurface" in key.lower():
        #         for obj in idf.idfobjects[key]:
        #             if hasattr(obj, "coords"):
        #                 building_coords.append(np.array(obj.coords)[:, :2])
        # buliding_centroid = np.concatenate(building_coords).mean(axis=0)

        # n_removed = 0
        # n_kept = 0
        # for key in idf.idfobjects:
        #     if "shading" in key.lower() and idf.idfobjects[key]:
        #         for obj in idf.idfobjects[key]:
        #             # remove the object from the idf
        #             if hasattr(obj, "coords"):
        #                 obj_coords = np.array(obj.coords)[:, :2]
        #                 obj_centroid = obj_coords.mean(axis=0)
        #                 if np.linalg.norm(obj_centroid - buliding_centroid) > (50):
        #                     idf.removeidfobject(obj)
        #                     n_removed += 1
        #                 else:
        #                     n_kept += 1
        # print(f"Removed {n_removed} shading surfaces")
        # print(f"Kept {n_kept} shading srfs")

        # extract metadata from the IDF
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
        results_file_path = results_dir / "eplusout.csv"
        hourly_results = extract_hourly_csv(results_file_path)
        hourly_results = hourly_results.drop(columns=["Date/Time"])

        polarities: list[Literal["overheating", "too_cold"]] = [
            "overheating",
            "too_cold",
        ]
        consecutive_hours: dict[str, float] = {}
        for threshold in [5, 10, 26, 30, 35]:
            for polarity in polarities:
                worst_case = compute_longest_consecutive_duration(
                    hourly_results, threshold, polarity
                )
                indicator = "C" if polarity == "too_cold" else "H"
                consecutive_hours[f"P_{indicator}_{threshold}"] = worst_case

        # convert the results to a dictionary of dataframes
        summary = compute_summary_stats(hourly_results)
        for key, value in consecutive_hours.items():
            summary[key] = value
        for azimuth, max_elevation in els_for_azimuth.items():
            summary[f"AZ_{azimuth}_max_elevation"] = float(max_elevation)

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
    """Normalize the zone name to a standard format.

    Generalize zone name normalization across typologies:
    e.g., 'FLR2_OFFICEBLDG_3_FLOOR1_ROOM2' → 'OfficeBldg_3_Floor2_Room2'
    """
    match = re.match(
        r"FLR(\d+)_([A-Z]+BLDG)_(\d+)_FLOOR\d+_ROOM(\d+)", zname, re.IGNORECASE
    )
    if match:
        floor_num, bldg_type, bldg_num, room_num = match.groups()
        # Building type
        bldg_type_formatted = bldg_type.capitalize()
        return f"{bldg_type_formatted}_{bldg_num}_Floor{floor_num}_Room{room_num}"

    # Fallback: return as-is if not matched
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
    rh_cols = [c for c in hourly_df.columns if "relative humidity" in c.lower()]

    if not temp_cols:
        return pd.Series({
            "peak_temp": None,
            "peak_temp_avg": None,
            "coldest_temp": None,
            "overheat_hours": None,
            "cold_hours": None,
            "avg_overheat_per_zone": None,  # new
            "hottest_zone": None,  # new
            "OH_26": None,
            "OH_30": None,
            "OH_35": None,
            "CH_10": None,
            "CH_5": None,
            "edh": None,
            "edh_hot": None,
            "edh_cold": None,
            "hi_extreme_danger": None,
            "hi_danger": None,
            "hi_extreme_caution": None,
            "hi_caution": None,
            "hi_normal": None,
        })

    hi_counts = heat_index_hourly_summary(hourly_df, temp_cols, rh_cols)
    hourly_df["max_T"] = hourly_df[temp_cols].max(axis=1)
    hourly_df["min_T"] = hourly_df[temp_cols].min(axis=1)

    # Basic stats
    peak_temp = hourly_df["max_T"].max()
    peak_temp_avg = hourly_df[temp_cols].max().mean()
    coldest_temp = hourly_df["min_T"].min()
    overheat_hours = (hourly_df["max_T"] > 26).sum()
    extreme_overheat_hours = (hourly_df["max_T"] > 30).sum()
    extreme_extreme_overheat_hours = (hourly_df["max_T"] > 35).sum()
    cold_hours = (hourly_df["min_T"] < 10).sum()
    extreme_cold_hours = (hourly_df["min_T"] < 5).sum()

    # New metrics added
    zone_oh_counts = [(hourly_df[c] > 26).sum() for c in temp_cols]
    avg_oh_per_zone = sum(zone_oh_counts) / len(temp_cols) if temp_cols else None
    hottest_zone_oh = max(zone_oh_counts) if temp_cols else None

    rep_zone_col = temp_cols[0]
    subset_df = cast(pd.DataFrame, hourly_df[[rep_zone_col]].copy())
    subset_df = subset_df.rename(columns={rep_zone_col: "Zone Air Temperature [C]"})
    if rh_cols:
        subset_df["Zone Air Relative Humidity [%]"] = hourly_df[rh_cols[0]]
    else:
        subset_df["Zone Air Relative Humidity [%]"] = 50.0

    total_edh_fixed, hot_edh_fixed, cold_edh_fixed = calculate_edh(hourly_df)

    return pd.Series({
        "peak_temp": round(peak_temp, 2),
        "peak_temp_avg": round(peak_temp_avg, 2),
        "coldest_temp": round(coldest_temp, 2),
        "avg_overheat_per_zone": round(avg_oh_per_zone, 2)
        if avg_oh_per_zone
        else None,  # new
        "hottest_zone": int(hottest_zone_oh) if hottest_zone_oh else None,  # new
        "OH_26": int(overheat_hours),
        "OH_30": int(extreme_overheat_hours),
        "OH_35": int(extreme_extreme_overheat_hours),
        "CH_10": int(cold_hours),
        "CH_5": int(extreme_cold_hours),
        "edh": round(total_edh_fixed, 2),
        "edh_hot": round(hot_edh_fixed, 2),
        "edh_cold": round(cold_edh_fixed, 2),
        "hi_extreme_danger": hi_counts["Extreme Danger"],
        "hi_danger": hi_counts["Danger"],
        "hi_extreme_caution": hi_counts["Extreme Caution"],
        "hi_caution": hi_counts["Caution"],
        "hi_normal": hi_counts["Normal"],
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


def compute_longest_consecutive_duration(
    hourly_results_columnar: pd.DataFrame,
    threshold: float,
    polarity: Literal["overheating", "too_cold"],
) -> int:
    """Compute the longest consecutive duration of a given polarity above or below a threshold."""
    mask = (
        hourly_results_columnar > threshold
        if polarity == "overheating"
        else hourly_results_columnar < threshold
    )
    cumsum = mask.cumsum()
    worst_cases_per_col = {}
    all_spans_per_col = {}
    ix_trackers = {}
    for col in cumsum.columns:
        if "temperature" not in col.lower():
            continue
        ix_tracker = []
        ix_trackers[col] = ix_tracker
        last_sum = 0
        state = "idle"
        count = 0
        consecutive_hours = []
        ix_start = None
        for ix, row in cumsum[col].to_dict().items():
            if state == "idle":
                if row > last_sum:
                    ix_start = ix
                    state = "counting"
                    count = 1
            elif state == "counting":
                if row > last_sum:
                    count += 1
                else:
                    state = "idle"
                    consecutive_hours.append(count)
                    ix_tracker.append(hourly_results_columnar[col].loc[ix_start:ix])
            last_sum = row
        worst_case = max(consecutive_hours) if (len(consecutive_hours) > 0) else 0
        is_ever_above_threshold = (cumsum[col] > threshold).any()
        is_ever_below_threshold = (cumsum[col] < threshold).any()
        if worst_case == 0 and (
            (is_ever_above_threshold and polarity == "overheating")
            or (is_ever_below_threshold and polarity == "too_cold")
        ):
            worst_case = 8760

        worst_cases_per_col[col] = worst_case
        all_spans_per_col[col] = consecutive_hours
    absolute_worst_case = max(worst_cases_per_col.values())

    return absolute_worst_case


def calculate_edh(df, met=1.1, clo=0.5, v=0.1, comfort_bounds=(22, 27)):
    """Calculates Exceedance Degree Hours (EDH) using fixed comfort bounds and separates hot and cold contributions.

    Parameters:
    - df: DataFrame with columns 'Zone Air Temperature [C]' and 'Zone Air Relative Humidity [%]'
    - met, clo, v: metabolic rate, clothing insulation, air speed
    - comfort_bounds: (lower, upper) SET range considered comfortable

    Returns:
    - total_edh, hot_edh, cold_edh (all rounded to 2 decimals)
    """
    from pythermalcomfort.models import set_tmp

    print("Calculating EDH with comfort range:", comfort_bounds)

    hot_edh = 0.0
    cold_edh = 0.0

    for _, row in df.iterrows():
        ta = row.get("Zone Air Temperature [C]", None)
        rh = row.get("Zone Air Relative Humidity [%]", 50.0)
        if ta is None:
            continue

        try:
            set_val = set_tmp(tdb=ta, tr=ta, rh=rh, v=v, met=met, clo=clo)["set"]
            lower, upper = comfort_bounds

            if set_val > upper:
                edh = set_val - upper
                hot_edh += edh
                print(f" HOT: SET={set_val:.2f} > {upper} → +{edh:.2f}")
            elif set_val < lower:
                edh = lower - set_val
                cold_edh += edh
                print(f"COLD: SET={set_val:.2f} < {lower} → +{edh:.2f}")
            # no else nee ded; if in comfort band → edh = 0

        except Exception as e:
            print(f" Error computing SET (TA={ta}, RH={rh}): {e}")
            continue

    total_edh = hot_edh + cold_edh
    return round(total_edh, 2), round(hot_edh, 2), round(cold_edh, 2)


def heat_index_hourly_summary(df, temp_cols, rh_cols):
    """Computes average heat index per hour across all zones and bins into 5 categories (sum = 8760).

    Uses Rothfusz regression and NOAA categories:
      - Extreme Danger: ≥130°F
      - Danger: 105 to 129°F
      - Extreme Caution: 90 to 104°F
      - Caution: 80 to 89°F
      - Normal: <80°F
    """

    def compute_category(temp_c, rh):
        # Convert to Fahrenheit
        temp_f = temp_c * 9 / 5 + 32
        hi_f = (
            -42.379
            + 2.04901523 * temp_f
            + 10.14333127 * rh
            - 0.22475541 * temp_f * rh
            - 6.83783e-3 * temp_f**2
            - 5.481717e-2 * rh**2
            + 1.22874e-3 * temp_f**2 * rh
            + 8.5282e-4 * temp_f * rh**2
            - 1.99e-6 * temp_f**2 * rh**2
        )

        if hi_f >= 130:
            return "Extreme Danger"
        elif 105 <= hi_f < 130:
            return "Danger"
        elif 90 <= hi_f < 105:
            return "Extreme Caution"
        elif 80 <= hi_f < 90:
            return "Caution"
        else:
            return "Normal"

    hi_hourly = []
    for t_col, rh_col in zip(temp_cols, rh_cols, strict=False):
        hi_zone = df[[t_col, rh_col]].apply(
            lambda row: compute_category(row[t_col], row[rh_col]),  # noqa: B023
            axis=1,
        )
        hi_hourly.append(hi_zone)

    # Get average category per hour by majority vote (or use first if tied)
    hi_avg = pd.concat(hi_hourly, axis=1).mode(axis=1)[0]
    counts = hi_avg.value_counts().to_dict()

    # Ensure all categories are present
    all_cats = ["Extreme Danger", "Danger", "Extreme Caution", "Caution", "Normal"]
    for cat in all_cats:
        counts.setdefault(cat, 0)

    return counts


def extract_surfaces(idf: IDF, filter_text: str = "shading") -> list[np.ndarray]:
    """Extract all surface coordinates from the IDF for the specified filter text."""
    all_coords = []
    for key in idf.idfobjects:
        if filter_text in key.lower() and idf.idfobjects[key]:
            for obj in idf.idfobjects[key]:
                coords = obj.coords
                all_coords.append(np.array(coords))
    return all_coords


def all_coords_to_pts(all_coords: list[np.ndarray]) -> np.ndarray:
    """Convert a list of geometry coordinates to a flattened list of points."""
    all_segments = []
    for coord in all_coords:
        for i in range(len(coord)):
            start = coord[i]
            end = coord[(i + 1) % len(coord)]
            midpoint = (start + end) / 2
            all_segments.append(np.array([start, midpoint, end]))
    return np.array(all_segments).reshape(-1, 3)


def get_azimuth_and_elevation(
    points: np.ndarray, source_point: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the azimuth and elevational angle of a segment which ends at each point and starts from a single point.

    Args:
        points (np.ndarray): shape is (N, 3), and represents terminus of a segment
        source_point (np.ndarray): shape is (3,), and represents the source point

    Returns:
        azimuth (np.ndarray): shape is (N,), and represents the azimuth of the segment
        elevation (np.ndarray): shape is (N,), and represents the angle of the segment
    """
    # Compute the azimuth and elevational angle of a segment which ends at each point and starts from a single point.
    azimuth = np.arctan2(points[:, 1] - source_point[1], points[:, 0] - source_point[0])
    elevation = np.arctan2(
        points[:, 2] - source_point[2],
        np.linalg.norm(points[:, :2] - source_point[:2], axis=1),
    )
    return azimuth, elevation


def round_azimuths_to_nearest(azimuths: np.ndarray, n_divs: float = 15) -> np.ndarray:
    """Round azimuths to the nearest multiple of an angle determined by the number of divisions."""
    rads_per_div = 2 * np.pi / (n_divs)
    return np.round(azimuths / rads_per_div) * rads_per_div


def compute_perimeter_of_regular_polygon(n_sides: int, apothem: float) -> float:
    """Compute the perimeter of a regular polygon with n_sides sides and a distance from the source point to the midpoint of the base edge of each vertical rectangle."""
    half_central_angle = np.pi / n_sides  # π / n
    perimeter = 2 * n_sides * apothem * np.tan(half_central_angle)  # 2 n a tan(π/n)

    return perimeter


def construct_shading_fence_coords(
    els_for_azimuth: pd.Series, source_point: np.ndarray, distance: float = 100
):
    """Construct a list of coordinates for rectangular surfaces which form a shading face in a circle around the source point.

    To construct each vertical rectangle in the shading fence, we start by moving the source point to a distance `distance` in the direction of the azimuth;
    this location now is the midpoint of the base edge of a surface which will be perpendicular to the azimuth and will extend the appropriate height verticallyfor the specified shading angle.


    Args:
        els_for_azimuth (pd.Series): shape is (N,), and represents the maximum elevation angle for each azimuth, where the azimuth is in the index.
        source_point (np.ndarray): shape is (3,), and represents the source point.
        distance (float): the distance from the source point to the midpoint of the base edge of each vertical rectangle; also necessary to determine the height of the vertical rectangles according to the specified elevation angle.

    Returns:
        coords (list[np.ndarray]): a list of coordinates for the vertices of the rectangular surfaces which form the shading fence; each array is of shape (4,3) and represents the coordinates of the vertices of a rectangle.
    """
    azimuths: np.ndarray = cast(np.ndarray, els_for_azimuth.index)
    elevation_angles: np.ndarray = cast(np.ndarray, els_for_azimuth.values)
    n_azimuths = len(azimuths)
    # compute the edge length for a regular polygon with n_azimuths sides and with distance from the source point to the midpoint of the base edge of each vertical rectangle
    edge_length = (
        compute_perimeter_of_regular_polygon(n_azimuths, distance) / n_azimuths
    )

    elevation_heights = distance * np.tan(elevation_angles)

    translated_source_points = (
        source_point
        + distance
        * np.array([np.cos(azimuths), np.sin(azimuths), np.zeros(len(azimuths))]).T
    )

    coords = []
    for azimuth, translated_source_point, elevation_height in zip(
        azimuths, translated_source_points, elevation_heights, strict=False
    ):
        perp_to_azimuth = np.array([
            -np.sin(azimuth),
            np.cos(azimuth),
            0,
        ])
        lower_left = translated_source_point + perp_to_azimuth * edge_length / 2
        lower_right = translated_source_point - perp_to_azimuth * edge_length / 2
        upper_left = lower_left + np.array([0, 0, elevation_height])
        upper_right = lower_right + np.array([0, 0, elevation_height])

        coords.append(np.array([lower_left, lower_right, upper_right, upper_left]))

    return coords


def create_shading_fence_coords(
    main_geo_coords: list[np.ndarray],
    shading_surface_coords: list[np.ndarray],
    n_divs: int = 36,
    distance: float = 100,
):
    """Create a list of coordinates for the vertices of the rectangular surfaces which form the shading fence.

    Args:
        main_geo_coords (list[np.ndarray]): a list of coordinates for the vertices of the main geometry.
        shading_surface_coords (list[np.ndarray]): a list of coordinates for the vertices of the shading surfaces.
        n_divs (int): the number of divisions to use for the azimuth.
        distance (float): the distance from the source point to the midpoint of the base edge of each vertical rectangle.

    Returns:
        coords (list[np.ndarray]): a list of coordinates for the vertices of the rectangular surfaces which form the shading fence.
        els_for_azimuth (pd.Series): shape is (N,), and represents the maximum elevation angle for each azimuth, where the azimuth is in the index.
    """
    all_main_pts = all_coords_to_pts(main_geo_coords)
    source_point = all_main_pts.mean(axis=0)
    source_point[-1] = 0

    all_shading_pts = all_coords_to_pts(shading_surface_coords)
    azimuth, el_angles = get_azimuth_and_elevation(all_shading_pts, source_point)
    azimuth_rounded = round_azimuths_to_nearest(azimuth, n_divs=n_divs)
    els_for_azimuth = (
        pd.DataFrame({
            "azimuth": azimuth,
            "azimuth_rounded": ((azimuth_rounded * 180 / np.pi) % 360) * np.pi / 180,
            "elevation": el_angles,
            "elevation_degrees": el_angles,
        })
        .groupby("azimuth_rounded")["elevation"]
        .max()
    )

    shading_fence_coords = construct_shading_fence_coords(
        cast(pd.Series, els_for_azimuth), source_point, distance=distance
    )
    return shading_fence_coords, els_for_azimuth


def remove_surfaces_from_idf(idf: IDF, filter_text: str = "shading"):
    """Remove all surfaces from the IDF which match the specified filter text."""
    for key in idf.idfobjects:
        if filter_text in key.lower() and idf.idfobjects[key]:
            for obj in idf.idfobjects[key]:
                idf.removeidfobject(obj)


def add_shading_fence_to_idf(idf: IDF, coords: list[np.ndarray]):
    """Add a list of coordinates for the vertices of the rectangular surfaces which form the shading fence to the IDF."""
    for i, coord in enumerate(coords):
        data = {
            "Name": f"Shading_Fence_{i:03d}",
            **{
                f"Vertex_{j}_Xcoordinate": float(coord[j % len(coord), 0])
                for j in range(len(coord) + 1)
            },
            **{
                f"Vertex_{j}_Ycoordinate": float(coord[j % len(coord), 1])
                for j in range(len(coord) + 1)
            },
            **{
                f"Vertex_{j}_Zcoordinate": float(coord[j % len(coord), 2])
                for j in range(len(coord) + 1)
            },
        }
        idf.newidfobject(
            "SHADING:SITE:DETAILED",
            **data,
        )


def simplify_shading(idf: IDF, n_divs: int = 36, distance: float = 100):
    """Simplify the shading in the IDF by removing the shading and replacing it with a shading fence."""
    shading_coords = extract_surfaces(idf, filter_text="shading")
    main_coords = extract_surfaces(idf, filter_text="buildingsurface")
    shading_fence_coords, els_for_azimuth = create_shading_fence_coords(
        main_coords, shading_coords, n_divs=n_divs, distance=distance
    )
    remove_surfaces_from_idf(idf, filter_text="shading")
    add_shading_fence_to_idf(idf, shading_fence_coords)
    return idf, els_for_azimuth
