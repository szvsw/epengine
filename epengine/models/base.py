"""Models for Simulation Specifications."""

import json
import logging
import tempfile
from pathlib import Path

from pydantic import AnyUrl, BaseModel, Field

from epengine.utils.filesys import fetch_uri

logger = logging.getLogger(__name__)


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
        if not path or path == "/":
            raise ValueError(f"URI:NO_PATH:{pth}")
        if path.startswith("/"):
            path = path[1:]
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


class LeafSpec(BaseSpec):
    """A spec for running a leaf workflow."""

    sort_index: int = Field(..., description="The sort index of the leaf.", ge=0)
