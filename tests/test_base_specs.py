"""Tests for the BaseSpec and LeafSpec classes."""

import importlib.resources as resources
import logging
from unittest.mock import patch

import pytest
from pydantic import AnyUrl, ValidationError

from epengine.models.base import BaseSpec, LeafSpec

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def test_basespec_initialization():
    """Test that BaseSpec can be initialized properly."""
    spec = BaseSpec(experiment_id="test_experiment")
    assert spec.experiment_id == "test_experiment"


def test_basespec_missing_experiment_id():
    """Test that BaseSpec raises an error if experiment_id is missing."""
    with pytest.raises(ValidationError):
        BaseSpec()


def test_leafspec_initialization():
    """Test that LeafSpec can be initialized properly."""
    spec = LeafSpec(experiment_id="test_experiment", sort_index=1)
    assert spec.experiment_id == "test_experiment"
    assert spec.sort_index == 1


def test_leafspec_sort_index_validation():
    """Test that LeafSpec raises an error if sort_index is negative."""
    with pytest.raises(ValidationError):
        LeafSpec(experiment_id="test_experiment", sort_index=-1)


def test_basespec_local_path():
    """Test the local_path method of BaseSpec."""
    spec = BaseSpec(experiment_id="test_experiment")
    uri = AnyUrl(url="http://example.com/path/to/file.txt")
    epengine_path = resources.files("epengine").parent
    expected_path = (
        epengine_path / "cache" / "test_experiment" / "path" / "to" / "file.txt"
    )
    local_path = spec.local_path(uri)
    assert local_path == expected_path


def test_basespec_local_path_no_path():
    """Test that local_path raises an error if uri has no path."""
    spec = BaseSpec(experiment_id="test_experiment")
    uri = AnyUrl(url="http://example.com")
    with pytest.raises(ValueError, match="URI:NO_PATH"):
        spec.local_path(uri)


def test_basespec_log(caplog):
    """Test the log method of BaseSpec."""
    spec = BaseSpec(experiment_id="test_experiment")
    with caplog.at_level(logging.INFO):
        spec.log("Test message")
    assert "Test message" in caplog.text


def test_basespec_fetch_uri():
    """Test the fetch_uri method of BaseSpec."""
    spec = BaseSpec(experiment_id="test_experiment")
    uri = AnyUrl(url="http://example.com/path/to/file.txt")
    epengine_path = resources.files("epengine").parent
    local_path = (
        epengine_path / "cache" / "test_experiment" / "path" / "to" / "file.txt"
    )
    # Mock the fetch_uri function
    with patch("epengine.models.base.fetch_uri") as mock_fetch_uri:
        mock_fetch_uri.return_value = local_path
        result = spec.fetch_uri(uri)
        assert result == local_path
        mock_fetch_uri.assert_called_once_with(uri, local_path, True, spec.log)


def test_basespec_fetch_uri_no_cache():
    """Test the fetch_uri method of BaseSpec with use_cache=False."""
    spec = BaseSpec(experiment_id="test_experiment")
    uri = AnyUrl(url="http://example.com/path/to/file.txt")
    epengine_path = resources.files("epengine").parent
    local_path = (
        epengine_path / "cache" / "test_experiment" / "path" / "to" / "file.txt"
    )
    with patch("epengine.models.base.fetch_uri") as mock_fetch_uri:
        mock_fetch_uri.return_value = local_path
        result = spec.fetch_uri(uri, use_cache=False)
        assert result == local_path
        mock_fetch_uri.assert_called_once_with(uri, local_path, False, spec.log)


def test_basespec_extra_fields():
    """Test that BaseSpec allows extra fields."""
    spec = BaseSpec(experiment_id="test_experiment", extra_field="extra_value")
    assert spec.experiment_id == "test_experiment"
    assert spec.extra_field == "extra_value"


def test_leafspec_extra_fields():
    """Test that LeafSpec allows extra fields."""
    spec = LeafSpec(
        experiment_id="test_experiment", sort_index=1, extra_field="extra_value"
    )
    assert spec.experiment_id == "test_experiment"
    assert spec.sort_index == 1
    assert spec.extra_field == "extra_value"
