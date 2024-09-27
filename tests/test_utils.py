"""Test the utility functions."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from epengine.utils.filesys import fetch_uri


def test_fetch_uri_http():
    """Test fetching a file from an HTTP URI."""
    # Test fetching a file from an HTTP URI
    with tempfile.TemporaryDirectory() as tempdir:
        local_path = Path(tempdir) / "testfile.txt"
        uri = "http://example.com/testfile.txt"
        content = b"Test content"

        # Mock requests.get
        with patch("requests.get") as mock_get:
            mock_get.return_value.content = content
            mock_get.return_value.status_code = 200

            # First fetch
            fetched_path = fetch_uri(uri, local_path)
            assert fetched_path == local_path
            assert local_path.exists()
            assert local_path.read_bytes() == content
            mock_get.assert_called_once_with(uri, timeout=60)

            # Fetch again with caching (should not call requests.get)
            mock_get.reset_mock()
            fetched_path = fetch_uri(uri, local_path)
            mock_get.assert_not_called()

            # Fetch again without caching
            fetched_path = fetch_uri(uri, local_path, use_cache=False)
            mock_get.assert_called_once_with(uri, timeout=60)


def test_fetch_uri_s3():
    """Test fetching a file from an S3 URI."""
    # Test fetching a file from an S3 URI
    with tempfile.TemporaryDirectory() as tempdir:
        local_path = Path(tempdir) / "testfile.txt"
        uri = "s3://mybucket/testfile.txt"
        content = b"Test content"

        # Mock boto3 S3 client
        with patch("boto3.client") as mock_boto3_client:
            mock_s3_client = MagicMock()
            mock_boto3_client.return_value = mock_s3_client

            def mock_download_file(Bucket, Key, Filename):
                with open(Filename, "wb") as f:
                    f.write(content)

            mock_s3_client.download_file.side_effect = mock_download_file

            # First fetch
            fetched_path = fetch_uri(uri, local_path, s3=mock_s3_client)
            assert fetched_path == local_path
            assert local_path.exists()
            assert local_path.read_bytes() == content
            mock_s3_client.download_file.assert_called_once_with(
                "mybucket", "testfile.txt", str(local_path)
            )

            # Fetch again with caching (should not call download_file)
            mock_s3_client.download_file.reset_mock()
            fetched_path = fetch_uri(uri, local_path, s3=mock_s3_client)
            mock_s3_client.download_file.assert_not_called()

            # Fetch again without caching
            fetched_path = fetch_uri(
                uri, local_path, use_cache=False, s3=mock_s3_client
            )
            mock_s3_client.download_file.assert_called_once_with(
                "mybucket", "testfile.txt", str(local_path)
            )


def test_fetch_uri_invalid_scheme():
    """Test fetching a file with an unsupported URI scheme."""
    # Test fetching a file with an unsupported URI scheme
    with tempfile.TemporaryDirectory() as tempdir:
        local_path = Path(tempdir) / "testfile.txt"
        uri = "ftp://example.com/testfile.txt"
        with pytest.raises(NotImplementedError) as exc_info:
            fetch_uri(uri, local_path)
        assert str(exc_info.value) == "URI:SCHEME:ftp"


def test_fetch_uri_no_path_s3():
    """Test fetching an S3 URI with no path."""
    # Test S3 URI with no path
    with tempfile.TemporaryDirectory() as tempdir:
        local_path = Path(tempdir) / "testfile.txt"
        uri = "s3://mybucket"
        with pytest.raises(ValueError) as exc_info:
            fetch_uri(uri, local_path)
        assert str(exc_info.value) == f"S3URI:NO_PATH:{uri}"


def test_fetch_uri_no_bucket_s3():
    """Test fetching an S3 URI with no bucket."""
    # Test S3 URI with no bucket
    with tempfile.TemporaryDirectory() as tempdir:
        local_path = Path(tempdir) / "testfile.txt"
        uri = "s3:///testfile.txt"
        with pytest.raises(ValueError) as exc_info:
            fetch_uri(uri, local_path)
        assert str(exc_info.value) == f"S3URI:NO_BUCKET:{uri}"
