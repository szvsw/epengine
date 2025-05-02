"""Workflows for the Tarkhan experiments."""

import asyncio
import re
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import boto3
import pandas as pd
from hatchet_sdk import Context
from pydantic import AnyUrl
from tqdm.autonotebook import tqdm

from epengine.experiments.tarkhan.models import TarkhanSpec, format_scoped_prefix
from epengine.hatchet import hatchet

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client
else:
    S3Client = Any


s3 = boto3.client("s3")


@hatchet.workflow(
    name="tarkhan",
    version="0.0.1",
    timeout="30m",
    schedule_timeout="300m",
)
class TarkhanWorkflow:
    """A workflow for running Tarkhan's simulation experiments."""

    @hatchet.step(
        name="simulate",
        timeout="30m",
        retries=0,
    )
    async def simulate(self, context: Context):
        """Simulate the Tarkhan experiments."""
        data = context.workflow_input()
        spec = TarkhanSpec(**data)

        results = await asyncio.to_thread(spec.run, s3=s3)
        return results


def make_experiment_specs(  # noqa: C901
    cases_zipfolder: Path,
    experiment_id: str,
    s3: S3Client,
    bucket: str = "ml-for-bem",
    bucket_prefix: str = "hatchet",
):
    """Make experiment specs from a zip file of cases."""
    # make sure cases_zipfolder is a zip file
    if cases_zipfolder.suffix != ".zip":
        msg = f"Cases zip folder {cases_zipfolder} is not a zip file"
        raise ValueError(msg)

    temp_dir = Path("local_artifacts").parent / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_dir) as tempdir:
        tempdir = Path(temp_dir)
        cases_folder = tempdir / cases_zipfolder.stem
        # unzip the cases_zipfolder
        print("Unzipping cases...")
        if not cases_folder.exists():
            with zipfile.ZipFile(cases_zipfolder, "r") as zip_ref:
                zip_ref.extractall(cases_folder)
        print("Done unzipping cases.")

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
        files_to_upload: list[tuple[str, str, str]] = []

        for subfolder in tqdm(
            matching_subfolders, desc="Subfolders", total=len(matching_subfolders)
        ):
            match = regex.match(subfolder.name)
            if match:
                case_id = match.group(1)
                city = match.group(2)
                grid_id = match.group(3)
                scoped_prefix = format_scoped_prefix(
                    bucket_prefix, experiment_id, case_id, city, grid_id
                )
                epws = list(subfolder.glob("*.epw"))
                if len(epws) != 1:
                    msg = f"Expected 1 epw file in {subfolder}, found {len(epws)}"
                    print(msg)
                    continue
                    # raise ValueError(msg)

                epw = epws[0]
                climate_regex = re.compile(r"^C\d+_([^_]+)_G\d+_(.*)\.epw")
                climate_matches = climate_regex.match(epw.name)
                if not climate_matches:
                    msg = f"Expected epw file to match regex {climate_regex}, but it did not"
                    raise ValueError(msg)
                # first match is city name
                # second match is the file type (rmy, tmy etc)
                epw_city = climate_matches.group(1)
                file_type = climate_matches.group(2)
                climate_key = f"{epw_city}_{file_type}"
                idfs = list(subfolder.glob("*.idf"))
                for idf in idfs:
                    idf_name = idf.stem
                    idf_key = f"{scoped_prefix}/{idf_name}.idf"
                    idf_uri = f"s3://{bucket}/{idf_key}"
                    epw_key = f"{scoped_prefix}/{epw.name}"
                    epw_uri = f"s3://{bucket}/{epw_key}"
                    if (idf.as_posix(), bucket, idf_key) not in files_to_upload:
                        files_to_upload.append((idf.as_posix(), bucket, idf_key))
                    if (epw.as_posix(), bucket, epw_key) not in files_to_upload:
                        files_to_upload.append((epw.as_posix(), bucket, epw_key))
                    spec = TarkhanSpec(
                        sort_index=0,
                        experiment_id=experiment_id,
                        idf_uri=AnyUrl(idf_uri),
                        epw_uri=AnyUrl(epw_uri),
                        case=case_id,
                        grid=grid_id,
                        climate=climate_key,
                        building_name=idf_name,
                        city=city,
                    )
                    all_specs.append(spec)
        df = pd.DataFrame([s.model_dump() for s in all_specs])
        df["sort_index"] = list(range(len(df)))
        keys = [key for _, _, key in files_to_upload]
        buckets = [bucket] * len(keys)
        filenames = [filename for filename, _, _ in files_to_upload]
        with ThreadPoolExecutor(max_workers=10) as executor:
            for _ in tqdm(
                executor.map(
                    s3.upload_file,
                    filenames,
                    buckets,
                    keys,
                ),
                desc="Uploading files",
                total=len(files_to_upload),
            ):
                pass
        return df


if __name__ == "__main__":
    import asyncio

    from hatchet_sdk import new_client

    client = new_client()
    experiment_id = "tarkhan/full-run"
    bucket = "ml-for-bem"
    bucket_prefix = "hatchet"
    recursion_map = {
        "factor": 100,
        "max_depth": 1,
    }
    # cases_zipfolder = Path(__file__).parent / "Cases.zip"
    # cases_zipfolder = Path("C:/users/szvsw/downloads/Cases.zip")
    cases_zipfolder = Path("local_artifacts") / "Cases.zip"
    res = make_experiment_specs(
        cases_zipfolder,
        experiment_id=experiment_id,
        s3=s3,
        bucket=bucket,
        bucket_prefix=bucket_prefix,
    )
    # res = res.sample(n=200)
    specs_key = f"{bucket_prefix}/{experiment_id}/specs.parquet"
    specs_uri = f"s3://{bucket}/{specs_key}"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        res.to_parquet(temp_dir / "res.parquet")
        s3.upload_file(
            (temp_dir / "res.parquet").as_posix(),
            bucket,
            specs_key,
        )

    payload = {
        "specs": specs_uri,
        "bucket": bucket,
        "workflow_name": "tarkhan",
        "experiment_id": experiment_id,
        "recursion_map": recursion_map,
    }
    workflow_ref = client.admin.run_workflow(
        "scatter_gather_recursive",
        input=payload,
    )
    print(f"Workflow run id: {workflow_ref.workflow_run_id}")
