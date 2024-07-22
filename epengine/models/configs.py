import logging
from functools import cached_property
from pathlib import Path

import boto3
import requests
from hatchet_sdk import Context
from pydantic import AnyUrl, BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

s3 = boto3.client("s3")


# TODO: should experiment ids be uuid trims?
# or should they be human readable (creating unicity issues...)?
# or should they also relate to hatchet auto-generated data?
class BaseSpec(BaseModel, arbitrary_types_allowed=True, extra="allow"):
    experiment_id: str = Field(..., description="The experiment_id of the spec")
    hcontext: Context | None = Field(
        None,
        description="The context of the spec when running in hatchet",
        exclude=True,
    )

    def local_path(self, pth: AnyUrl) -> Path:
        """
        Return the local path of a uri scoped to the experiment_id.

        Args:
            pth (AnyUrl): The uri to convert to a local path

        Returns:
            local_path (Path): The local path of the uri
        """
        path = pth.path
        return Path("/local_artifacts") / self.experiment_id / path

    def log(self, msg: str):
        """
        Log a message to the context or to the logger.

        Args:
            msg (str): The message to log
        """
        if self.hcontext:
            self.hcontext.log(msg)
        else:
            logger.info(msg)

    def fetch_uri(self, uri: AnyUrl) -> Path:
        """
        Fetch a file from a uri and return the local path.

        Args:
            uri (AnyUrl): The uri to fetch

        Returns:
            local_path (Path): The local path of the fetched file
        """
        local_path = self.local_path(uri)
        if uri.scheme == "s3":
            bucket = uri.host
            path = uri.path[1:]
            if not local_path.exists():
                self.log(f"Downloading {uri}...")
                local_path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, path, str(local_path))
            else:
                self.log(f"File {local_path} already exists, skipping download.")
        elif uri.scheme == "file":
            if not local_path.exists():
                self.log(f"Copying {uri}...")
                local_path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(uri.host, uri.path, str(local_path))
            else:
                self.log(f"File {local_path} already exists, skipping copy.")
        elif uri.scheme == "http" or uri.scheme == "https":
            if not local_path.exists():
                self.log(f"Downloading {uri}...")
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, "wb") as f:
                    f.write(requests.get(uri, timeout=60).content)
            else:
                self.log(f"File {local_path} already exists, skipping download.")
        return local_path


class SimulationSpec(BaseSpec):
    idf_uri: AnyUrl = Field(..., description="The uri of the idf file to fetch and simulate")
    epw_uri: AnyUrl = Field(..., description="The uri of the epw file to fetch and simulate")

    @cached_property
    def idf_path(self):
        """
        Fetch the idf file and return the local path.
        """
        return self.fetch_uri(self.idf_uri)

    @cached_property
    def epw_path(self):
        """
        Fetch the epw file and return the local path.
        """
        return self.fetch_uri(self.epw_uri)


class SimulationsSpec(BaseSpec):
    specs: list[SimulationSpec] = Field(..., description="The list of simulation specs to run")

    @model_validator(mode="before")
    @classmethod
    def set_children_experiment_id(cls, values):
        """
        Set the experiment_id of each child spec to the experiment_id of the parent.
        """
        for spec in values["specs"]:
            spec["experiment_id"] = values["experiment_id"]
        return values

    # TODO: allow passing specs_uri and retrieving, e.g. a json file
