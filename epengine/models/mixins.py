"""Mixin Classes for working with the epengine models."""

from hatchet_sdk.v0 import Context
from pydantic import BaseModel, Field


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

    bucket: str = Field(..., description="The bucket to store the results")


class WithOptionalBucket(BaseModel):
    """A model with an optional bucket to store results."""

    bucket: str | None = Field(
        default=None, description="The bucket to store the results"
    )
