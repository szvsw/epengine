"""Filesystem utilities."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
import requests
from pydantic import AnyUrl

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client as S3ClientType
else:
    S3ClientType = object

logger = logging.getLogger(__name__)
s3: S3ClientType = boto3.client("s3")


def fetch_uri(
    uri: AnyUrl | str,
    local_path: Path,
    use_cache: bool = True,
    logger_fn: Callable = logger.info,
    s3: S3ClientType = s3,
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
        s3 (S3Client): The S3 client to use

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
