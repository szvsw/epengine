import json
import logging
import tempfile
from collections.abc import Callable
from functools import cached_property
from pathlib import Path

import boto3
import requests
from hatchet_sdk import Context
from pydantic import AnyUrl, BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

s3 = boto3.client("s3")


def fetch_uri(
    uri: AnyUrl | str,
    local_path: Path,
    use_cache: bool = True,
    logger_fn: Callable = logger.info,
) -> Path:
    """Fetch a file from a uri and return the local path.

    Caching is enabled by default and works by
    checking if the file exists locally before downloading it
    to avoid downloading the same file multiple times.

    Args:
        uri (AnyUrl): The uri to fetch
        local_path (Path): The local path to save the fetched file
        use_cache (bool): Whether to use the cache
        logger_fn (Callable): The logger function to use

    Returns:
        local_path (Path): The local path of the fetched file
    """
    if isinstance(uri, str):
        uri = AnyUrl(uri)
    if uri.scheme == "s3":
        bucket = uri.host
        if not uri.path:
            raise ValueError(f"S3URI:NO_PATH:{uri}")
        if not bucket:
            raise ValueError(f"S3URI:NO_BUCKET:{uri}")
        path = uri.path[1:]
        if not local_path.exists() or not use_cache:
            logger_fn(f"Downloading {uri}...")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, path, str(local_path))
        else:
            logger_fn(f"File {local_path} already exists, skipping download.")
    elif uri.scheme == "http" or uri.scheme == "https":
        if not local_path.exists() or not use_cache:
            logger_fn(f"Downloading {uri}...")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(requests.get(str(uri), timeout=60).content)
        else:
            logger_fn(f"File {local_path} already exists, skipping download.")
    else:
        raise NotImplementedError(f"URI:SCHEME:{uri.scheme}")
    return local_path


# TODO: should experiment ids be uuid trims?
# or should they be human readable (creating unicity issues...)?
# or should they also relate to hatchet auto-generated data?
class BaseSpec(BaseModel, extra="allow", arbitrary_types_allowed=True):
    """A base spec for running a simulation.

    The main features are utilities to fetch files from uris
    and generate a locally scoped path for the files.
    according to the experiment_id.
    """

    experiment_id: str = Field(..., description="The experiment_id of the spec")

    def local_path(self, pth: AnyUrl) -> Path:
        """Return the local path of a uri scoped to the experiment_id.

        Note that this should only be used for non-ephemeral files.

        Args:
            pth (AnyUrl): The uri to convert to a local path

        Returns:
            local_path (Path): The local path of the uri
        """
        path = pth.path
        if not path:
            raise ValueError(f"URI:NO_PATH:{pth}")
        return Path("/local_artifacts") / self.experiment_id / path

    def log(self, msg: str):
        """Log a message to the context or to the logger.

        Args:
            msg (str): The message to log
        """
        logger.info(msg)

    def fetch_uri(self, uri: AnyUrl, use_cache: bool = True) -> Path:
        """Fetch a file from a uri and return the local path.

        Args:
            uri (AnyUrl): The uri to fetch
            use_cache (bool): Whether to use the cache

        Returns:
            local_path (Path): The local path of the fetched file
        """
        local_path = self.local_path(uri)
        return fetch_uri(uri, local_path, use_cache, self.log)

    @classmethod
    def from_uri(cls, uri: AnyUrl | str):
        """Fetch a spec from a uri and return the spec.

        Args:
            uri (AnyUrl): The uri to fetch

        Returns:
            spec (BaseSpec): The fetched spec
        """

        if isinstance(uri, str):
            uri = AnyUrl(uri)

        if Path(str(uri)).suffix != ".json":
            raise NotImplementedError("URI:SUFFIX:JSON_ONLY")

        with tempfile.TemporaryDirectory() as tempdir:
            local_path = Path(tempdir) / Path(str(uri)).name
            local_path = fetch_uri(uri, local_path, use_cache=False)
            with open(local_path) as f:
                spec_data = json.load(f)
                return cls(**spec_data)

    @classmethod
    def from_payload(cls, payload: dict):
        """Create a simulation spec from a payload.

        Fetches a spec from a payload and return the spec,
        or validates it if it is already a spec dict.

        Args:
            payload (dict): The payload to fetch

        Returns:
            spec (BaseSpec): The fetched spec
        """
        if "uri" in payload:
            return cls.from_uri(payload["uri"])
        else:
            return cls(**payload)


class SimulationSpec(BaseSpec):
    """A spec for running an EnergyPlus simulation."""

    idf_uri: AnyUrl = Field(..., description="The uri of the idf file to fetch and simulate")
    epw_uri: AnyUrl = Field(..., description="The uri of the epw file to fetch and simulate")
    ddy_uri: AnyUrl | None = Field(None, description="The uri of the ddy file to fetch and simulate")

    @cached_property
    def idf_path(self):
        """Fetch the idf file and return the local path.

        Returns:
            local_path (Path): The local path of the idf file
        """
        return self.fetch_uri(self.idf_uri)

    @cached_property
    def epw_path(self):
        """Fetch the epw file and return the local path.

        Returns:
            local_path (Path): The local path of the epw file
        """
        return self.fetch_uri(self.epw_uri)

    @cached_property
    def ddy_path(self):
        """Fetch the ddy file and return the local path.

        Returns:
            local_path (Path): The local path of the ddy file
        """
        if self.ddy_uri:
            return self.fetch_uri(self.ddy_uri)
        return None


class RecursionSpec(BaseModel):
    """A spec for recursive calls."""

    factor: int = Field(..., description="The factor to use in recursive calls", ge=1)
    offset: int | None = Field(default=None, description="The offset to use in recursive calls", ge=0)

    @model_validator(mode="before")
    @classmethod
    def validate_offset_less_than_factor(cls, values):
        """
        Validate that the offset is less than the factor.
        """
        if values["offset"] is None:
            return values
        if values["offset"] >= values["factor"]:
            raise ValueError(f"OFFSET:{values['offset']}>=FACTOR:{values['factor']}")
        return values


class RecursionMap(BaseModel):
    """A map of recursion specs to use in recursive calls.

    This allows a recursion node to understand where
    it is in the recursion tree and how to behave.
    """

    path: list[RecursionSpec] | None = Field(default=None, description="The path of recursion specs to use")
    factor: int = Field(..., description="The factor to use in recursive calls", ge=1)

    @field_validator("path", mode="before")
    @classmethod
    def validate_path_is_length_ge_1(cls, values):
        """
        Validate that the path is at least length 1.
        """
        if values is None:
            return values
        if len(values) < 1:
            raise ValueError("PATH:LENGTH:GE:1")
        return values


class WithHContext(BaseModel, extra="allow", arbitrary_types_allowed=True):
    """A model with a Hatchet context."""

    hcontext: Context = Field(
        ...,
        description="The context of the spec when running in hatchet",
        exclude=True,
    )

    def log(self, msg: str):
        """Log a message to the hatchet context.

        Args:
            msg (str): The message to log
        """
        self.hcontext.log(msg)


class WithBucket(BaseModel):
    """A model with a bucket to store results."""

    bucket: str = Field(default=None, description="The bucket to store the results")


class WithOptionalBucket(BaseModel):
    """A model with an optional bucket to store results."""

    bucket: str | None = Field(default=None, description="The bucket to store the results")


class SimulationsSpec(BaseSpec):
    """A spec for running multiple simulations.

    One key feature is that children simulations
    inherit the experiment_id of the parent simulations BaseSpec
    since they are both part of the same experiment.
    """

    specs: list[SimulationSpec] = Field(..., description="The list of simulation specs to run")

    @model_validator(mode="before")
    @classmethod
    def set_children_experiment_id(cls, values: dict):
        """Set the experiment_id of each child spec to the experiment_id of the parent."""

        if values.get("specs") is not None:
            for spec in values["specs"]:
                if isinstance(spec, dict):
                    spec["experiment_id"] = values["experiment_id"]
                elif isinstance(spec, SimulationSpec):
                    spec.experiment_id = values["experiment_id"]
                else:
                    raise TypeError(f"SPEC:TYPE:{type(spec)}")

        return values


class URIResponse(BaseModel):
    """A response containing the uri of a file"""

    uri: AnyUrl = Field(..., description="The uri of file")
