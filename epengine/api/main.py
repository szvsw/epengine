import json
import logging
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import boto3
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from hatchet_sdk import new_client
from tqdm import tqdm

api = FastAPI()
client = new_client()

# Make clients
s3 = boto3.client("s3")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@api.get("/")
def root():
    return {"Hello": "World"}


@api.post("/simulate-artifacts")
def simulate_artifacts(  # noqa: C901
    experiment_id: str,
    epws: UploadFile = File(...),  # noqa: B008
    idfs: UploadFile = File(...),  # noqa: B008
    config: UploadFile = File(...),  # noqa: B008
    bucket: str = "ml-for-bem",
    bucket_prefix: str = "hatchet",
):
    job_id = str(uuid4())
    remote_root = f"{bucket_prefix}/{experiment_id}/{job_id[:8]}"

    def format_path(folder, key):
        return f"{remote_root}/{folder}/{key}"

    def format_s3_path(folder, key):
        return f"s3://{bucket}/{format_path(folder, key)}"

    logger.info(f"Loading config file {config.filename}...")
    with open(config.file) as f:
        config_data = json.load(f)
    logger.info(f"Config file {config.filename} loaded.")
    df: pd.DataFrame = pd.DataFrame.from_dict(config_data, orient="records")

    if "epw_path" not in df.columns and "epw_generator" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="epw_path or epw_generator column required in config dataframe.",
        )
    if "epw_path" in df.columns and "epw_generator" in df.columns:
        raise HTTPException(
            status_code=400,
            detail="epw_path and epw_generator columns cannot both be present in config dataframe.",
        )

    if "idf_path" not in df.columns and "idf_generator" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="idf_path or idf_generator column required in config dataframe.",
        )

    if "idf_path" in df.columns and "idf_generator" in df.columns:
        raise HTTPException(
            status_code=400,
            detail="idf_path and idf_generator columns cannot both be present in config dataframe.",
        )

    uploads_epws = "epw_path" in df.columns
    uploads_idfs = "idf_path" in df.columns

    def upload_to_s3(destination_key: str, source_path: str):
        s3.upload_file(
            Filename=source_path,
            Bucket=bucket,
            Key=destination_key,
        )

    if uploads_epws:
        with tempfile.TemporaryDirectory() as tempdir:
            with open(Path(tempdir) / epws.filename, "wb") as f:
                f.write(epws.file.read())
            shutil.unpack_archive(Path(tempdir) / epws.filename, tempdir)
            local_epw_paths: list[Path] = list(Path(tempdir).rglob("*.epw"))
            epw_paths_to_upload: list[str] = df.epw_path.unique().tolist()
            if not set(epw_paths_to_upload).issubset({path.name for path in local_epw_paths}):
                raise HTTPException(
                    status_code=400,
                    detail="Not all epws listed in config dataframe are present in epws.zip.",
                )
            df["epw_uri"] = df.epw_path.apply(lambda x: format_s3_path("epw", Path(x).name))
            df.pop("epw_path")
            epw_paths_to_upload = [(Path(tempdir) / path).as_posix() for path in epw_paths_to_upload]
            epw_path_destinations = [format_path("epw", Path(path).name) for path in epw_paths_to_upload]
            with ThreadPoolExecutor(max_workers=10) as executor:
                list(
                    tqdm(
                        executor.map(upload_to_s3, epw_path_destinations, epw_paths_to_upload),
                        total=len(epw_paths_to_upload),
                        desc="Uploading epws to s3...",
                    )
                )

    if uploads_idfs:
        with tempfile.TemporaryDirectory() as tempdir:
            with open(Path(tempdir) / idfs.filename, "wb") as f:
                f.write(idfs.file.read())
            shutil.unpack_archive(Path(tempdir) / idfs.filename, tempdir)
            local_idf_paths: list[Path] = list(Path(tempdir).rglob("*.idf"))
            idf_paths_to_upload: list[str] = df.idf_path.unique().tolist()
            if not set(idf_paths_to_upload).issubset({path.name for path in local_idf_paths}):
                raise HTTPException(
                    status_code=400,
                    detail="Not all idfs listed in config dataframe are present in idfs.zip.",
                )
            df["idf_uri"] = df.idf_path.apply(lambda x: format_s3_path("idf", Path(x).name))
            df.pop("idf_path")
            idf_paths_to_upload = [(Path(tempdir) / path).as_posix() for path in idf_paths_to_upload]
            idf_path_destinations = [format_path("idf", Path(path).name) for path in idf_paths_to_upload]
            with ThreadPoolExecutor(max_workers=10) as executor:
                list(
                    tqdm(
                        executor.map(upload_to_s3, idf_path_destinations, idf_paths_to_upload),
                        total=len(idf_paths_to_upload),
                        desc="Uploading idfs to s3...",
                    )
                )

    config_dict = df.to_dict(orient="records")
    with tempfile.TemporaryDirectory() as tempdir:
        with open(Path(tempdir) / "config.json", "w") as f:
            json.dump(config_dict, f)
        config_path = Path(tempdir) / "config.json"
        upload_to_s3(format_path("config", "config.json"), config_path.as_posix())

    config_uri = format_s3_path("config", "config.json")

    payload = {
        "specs": config_dict,
        "config_uri": config_uri,
        "experiment_id": experiment_id,
        "bucket": bucket,
        "job_id": job_id,
        "workflow_run_id": None,
    }

    workflow_run_id = client.admin.run_workflow(
        workflow_name="scatter_gather",
        input=payload,
    )
    return {"workflow_run_id": workflow_run_id, "job_id": job_id}


@api.get("/workflows")
def get_workflows():
    return [row.name for row in client.rest.workflow_list().rows]
