"""Models for workflow outputs etc."""

from pydantic import AnyUrl, BaseModel, Field


class URIResponse(BaseModel):
    """A response containing the uri of a file."""

    uri: AnyUrl = Field(..., description="The uri of file")
