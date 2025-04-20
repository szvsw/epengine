"""Models for manifests."""

from collections.abc import Sequence

from pydantic import BaseModel

from epengine.gis.models import GisJobArgs


class Manifest(BaseModel):
    """A manifest for a sequence of jobs."""

    Name: str
    Jobs: Sequence[GisJobArgs]
