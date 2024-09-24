"""Main API module for the EP Engine."""

import json
import logging
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import boto3
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from hatchet_sdk import new_client
from tqdm import tqdm

from epengine.models.configs import SimulationsSpec, fetch_uri

api = FastAPI()
client = new_client()

# Make clients
s3 = boto3.client("s3")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)


@api.get("/")
def root():
    """Root endpoint."""
    return {"Hello": "World"}


@api.post("/simulate-artifacts")
async def simulate_artifacts(  # noqa: C901
    experiment_id: str,
    epws: UploadFile = File(...),  # noqa: B008
    idfs: UploadFile = File(...),  # noqa: B008
    specs: UploadFile = File(...),  # noqa: B008
    ddys: UploadFile = File(...),  # noqa: B008
    bucket: str = "ml-for-bem",
    bucket_prefix: str = "hatchet",
    existing_artifacts: Literal["overwrite", "forbid"] = "forbid",
    recursion_factor: int | None = 4,
):
    """An endpoint to schedule simulation of EnergyPlus artifacts.

    This endpoint will do the following:
    1. Upload the epw, idf, specs and ddy files to s3.
    2. Schedule the simulation of the EnergyPlus artifacts.

    Args:
        experiment_id (str): The experiment id.
        epws (UploadFile): The epw zip file.
        idfs (UploadFile): The idf zip file.
        specs (UploadFile): The specs json file.
        ddys (UploadFile): The ddy zip file.
        bucket (str, optional): The bucket to use. Defaults to "ml-for-bem".
        bucket_prefix (str, optional): The bucket prefix. Defaults to "hatchet".
        existing_artifacts (Literal["overwrite", "forbid"], optional): Whether to overwrite existing artifacts. Defaults to "forbid".
        recursion_factor (int | None, optional): The recursion factor. Defaults to 4.

    Returns:
        dict: The workflow run id and the number of jobs.
    """
    remote_root = f"{bucket_prefix}/{experiment_id}"

    def format_path(folder, key):
        return f"{remote_root}/{folder}/{key}"

    def format_s3_path(folder, key):
        return f"s3://{bucket}/{format_path(folder, key)}"

    # check if experiment_id already exists
    if (
        s3.list_objects_v2(Bucket=bucket, Prefix=remote_root).get("KeyCount", 0) > 0
        and existing_artifacts == "forbid"
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Experiment '{experiment_id}' already exists. Set 'existing_artifacts' to 'overwrite' to confirm.",
        )

    logger.info(f"Loading config file {specs.filename}...")
    contents = await specs.read()
    spec_data = json.loads(contents.decode("utf-8"))
    logger.info(f"Config file {specs.filename} loaded.")
    df: pd.DataFrame = pd.DataFrame(spec_data)

    if "epw_path" not in df.columns and "epw_generator" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="epw_path or epw_generator column required in specs dataframe.",
        )
    if "epw_path" in df.columns and "epw_generator" in df.columns:
        raise HTTPException(
            status_code=400,
            detail="epw_path and epw_generator columns cannot both be present in specs dataframe.",
        )

    if "idf_path" not in df.columns and "idf_generator" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="idf_path or idf_generator column required in specs dataframe.",
        )

    if "idf_path" in df.columns and "idf_generator" in df.columns:
        raise HTTPException(
            status_code=400,
            detail="idf_path and idf_generator columns cannot both be present in specs dataframe.",
        )

    uploads_epws = "epw_path" in df.columns
    uploads_idfs = "idf_path" in df.columns
    upload_ddys = "ddy_path" in df.columns

    def upload_to_s3(destination_key: str, source_path: str):
        s3.upload_file(
            Filename=source_path,
            Bucket=bucket,
            Key=destination_key,
        )

    if uploads_epws:
        with tempfile.TemporaryDirectory() as tempdir:
            epw_local_path = Path(tempdir) / (epws.filename or "epw.zip")
            with open(epw_local_path, "wb") as f:
                f.write(epws.file.read())
            shutil.unpack_archive(epw_local_path, tempdir)
            local_epw_paths: list[Path] = list(Path(tempdir).rglob("*.epw"))
            epw_paths_to_upload: list[str] = df.epw_path.unique().tolist()
            if not set(epw_paths_to_upload).issubset({
                path.name for path in local_epw_paths
            }):
                raise HTTPException(
                    status_code=400,
                    detail="Not all epws listed in config dataframe are present in epws.zip.",
                )
            df["epw_uri"] = df.epw_path.apply(
                lambda x: format_s3_path("epw", Path(x).name)
            )
            df.pop("epw_path")
            epw_paths_to_upload = [
                (Path(tempdir) / path).as_posix() for path in epw_paths_to_upload
            ]
            epw_path_destinations = [
                format_path("epw", Path(path).name) for path in epw_paths_to_upload
            ]
            logger.info("Uploading epws to s3...")
            with ThreadPoolExecutor(max_workers=10) as executor:
                list(
                    tqdm(
                        executor.map(
                            upload_to_s3, epw_path_destinations, epw_paths_to_upload
                        ),
                        total=len(epw_paths_to_upload),
                        desc="Uploading epws to s3...",
                    )
                )

    if uploads_idfs:
        with tempfile.TemporaryDirectory() as tempdir:
            idf_local_path = Path(tempdir) / (idfs.filename or "idf.zip")
            with open(idf_local_path, "wb") as f:
                f.write(idfs.file.read())
            shutil.unpack_archive(idf_local_path, tempdir)
            local_idf_paths: list[Path] = list(Path(tempdir).rglob("*.idf"))
            idf_paths_to_upload: list[str] = df.idf_path.unique().tolist()
            if not set(idf_paths_to_upload).issubset({
                path.name for path in local_idf_paths
            }):
                raise HTTPException(
                    status_code=400,
                    detail="Not all idfs listed in specs dataframe are present in idfs.zip.",
                )
            df["idf_uri"] = df.idf_path.apply(
                lambda x: format_s3_path("idf", Path(x).name)
            )
            df.pop("idf_path")
            idf_paths_to_upload = [
                (Path(tempdir) / path).as_posix() for path in idf_paths_to_upload
            ]
            idf_path_destinations = [
                format_path("idf", Path(path).name) for path in idf_paths_to_upload
            ]
            with ThreadPoolExecutor(max_workers=10) as executor:
                list(
                    tqdm(
                        executor.map(
                            upload_to_s3, idf_path_destinations, idf_paths_to_upload
                        ),
                        total=len(idf_paths_to_upload),
                        desc="Uploading idfs to s3...",
                    )
                )

    if upload_ddys:
        with tempfile.TemporaryDirectory() as tempdir:
            ddy_local_path = Path(tempdir) / (ddys.filename or "ddy.zip")
            with open(ddy_local_path, "wb") as f:
                f.write(ddys.file.read())
            shutil.unpack_archive(ddy_local_path, tempdir)
            local_ddy_paths: list[Path] = list(Path(tempdir).rglob("*.ddy"))
            ddy_paths_to_upload: list[str] = df.ddy_path.unique().tolist()
            if not set(ddy_paths_to_upload).issubset({
                path.name for path in local_ddy_paths
            }):
                raise HTTPException(
                    status_code=400,
                    detail="Not all ddys listed in specs dataframe are present in ddys.zip.",
                )
            df["ddy_uri"] = df.ddy_path.apply(
                lambda x: format_s3_path("ddy", Path(x).name)
            )
            df.pop("ddy_path")
            ddy_paths_to_upload = [
                (Path(tempdir) / path).as_posix() for path in ddy_paths_to_upload
            ]
            ddy_path_destinations = [
                format_path("ddy", Path(path).name) for path in ddy_paths_to_upload
            ]
            with ThreadPoolExecutor(max_workers=10) as executor:
                list(
                    tqdm(
                        executor.map(
                            upload_to_s3, ddy_path_destinations, ddy_paths_to_upload
                        ),
                        total=len(ddy_paths_to_upload),
                        desc="Uploading ddys to s3...",
                    )
                )

    df["sort_index"] = list(range(len(df)))
    specs_dict = df.to_dict(orient="records")

    specs_uri = format_s3_path("specs", "specs.json")

    payload = {
        "specs": specs_dict,
        "experiment_id": experiment_id,
        "bucket": bucket,
    }

    # validate the payload
    SimulationsSpec(**payload.copy())

    # upload the specs to s3
    with tempfile.TemporaryDirectory() as tempdir:
        with open(Path(tempdir) / "specs.json", "w") as f:
            json.dump(payload, f)
        specs_path = Path(tempdir) / "specs.json"
        upload_to_s3(format_path("specs", "specs.json"), specs_path.as_posix())

    workflow_payload: dict[str, Any] = {
        "uri": specs_uri,
    }

    if recursion_factor is not None:
        workflow_payload["recursion_map"] = {"factor": recursion_factor}

    workflowRef = client.admin.run_workflow(
        workflow_name="scatter_gather"
        if recursion_factor is None
        else "scatter_gather_recursive",
        input=workflow_payload,
    )
    return {"workflow_run_id": workflowRef.workflow_run_id, "n_jobs": len(specs_dict)}


@api.get("/workflows/{workflow_run_id}")
async def get_workflow(workflow_run_id: str, bg_tasks: BackgroundTasks) -> FileResponse:
    """Get the results of a workflow run.

    Args:
        workflow_run_id (str): The workflow run id.
        bg_tasks (BackgroundTasks): The background tasks.

    Returns:
        FileResponse: The results of the workflow run.
    """
    workflow = client.admin.get_workflow_run(workflow_run_id)
    res = await workflow.result()
    if "spawn_children" in res:
        data = res["spawn_children"]
        # TODO: why are we doing it this way with a bg task
        # for removing the tmpdir?  why not use a context
        # manager?
        tempdir = tempfile.mkdtemp()

        local_path = Path(tempdir) / "results.h5"
        if "uri" in data:
            local_path = fetch_uri(data["uri"], local_path, use_cache=False)
        else:
            for key, df_dict in data.items():
                df = pd.DataFrame.from_dict(df_dict, orient="tight")
                df.to_hdf(local_path, key=key, mode="a")
        bg_tasks.add_task(shutil.rmtree, tempdir)
        return FileResponse(
            local_path.as_posix(),
            media_type="application/octet-stream",
            filename=f"{workflow_run_id[:8]}.h5",
            background=bg_tasks,
        )

    else:
        raise HTTPException(status_code=404, detail="Results not found")


@api.get("/workflows")
def get_workflows():
    """Get a list of workflows.

    Returns:
        list: The list of workflows.
    """
    return [row.name for row in client.rest.workflow_list().rows or []]
