"""Tests for the BranchesSpec class."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from epengine.models.branches import BranchesSpec
from epengine.models.leafs import SimpleSpec


def test_branchesspec_with_thick_specs():
    """Test BranchesSpec with directly provided specs (thick payload)."""
    specs = [
        SimpleSpec(experiment_id="test", sort_index=i, param_a=i).model_dump()
        for i in range(3)
    ]
    branch_spec = BranchesSpec(experiment_id="test", specs=specs)
    assert len(branch_spec.specs) == 3
    assert all(spec.experiment_id == "test" for spec in branch_spec.specs)
    assert [spec.param_a for spec in branch_spec.specs] == [0, 1, 2]


def test_branchesspec_with_thin_specs_parquet():
    """Test BranchesSpec with specs provided as a URI to a parquet file (thin payload)."""
    # Create test data
    specs = [
        SimpleSpec(experiment_id="original", sort_index=i, param_a=i).model_dump()
        for i in range(3)
    ]
    df = pd.DataFrame(specs)

    with (
        tempfile.TemporaryDirectory() as tempdir,
        patch("epengine.models.branches.fetch_uri") as mock_fetch_uri,
    ):
        # Save test data to parquet
        specs_path = Path(tempdir) / "specs.pq"
        df.to_parquet(specs_path)
        mock_fetch_uri.return_value = specs_path

        # Test loading from URI
        branch_spec = BranchesSpec(experiment_id="test", specs="s3://bucket/specs.pq")

        # Check that specs were loaded and experiment_id was properly set
        assert len(branch_spec.specs) == 3
        assert all(spec.experiment_id == "test" for spec in branch_spec.specs)
        assert [spec.param_a for spec in branch_spec.specs] == [0, 1, 2]
        assert [spec.sort_index for spec in branch_spec.specs] == [0, 1, 2]


def test_branchesspec_with_invalid_uri_ext():
    """Test BranchesSpec with an invalid URI that doesn't point to a parquet or JSON file."""
    with (
        tempfile.TemporaryDirectory() as tempdir,
        patch("epengine.models.branches.fetch_uri") as mock_fetch_uri,
    ):
        # Create an invalid file (not parquet or JSON)
        specs_path = Path(tempdir) / "specs.txt"
        specs_path.write_text("invalid data")
        mock_fetch_uri.return_value = specs_path

        with pytest.raises(ValueError, match="SPEC:URI:EXT:.txt"):
            BranchesSpec(experiment_id="test", specs="s3://bucket/specs.txt")


def test_branchesspec_with_empty_specs():
    """Test BranchesSpec with an empty specs list."""
    branch_spec = BranchesSpec(experiment_id="test", specs=[])
    assert len(branch_spec.specs) == 0


def test_branchesspec_missing_experiment_id():
    """Test that BranchesSpec raises an error if experiment_id is missing."""
    specs = [SimpleSpec(experiment_id="test", sort_index=0, param_a=0).model_dump()]
    with pytest.raises(KeyError, match="experiment_id"):
        BranchesSpec(specs=specs)


def test_branchesspec_deser_and_set_exp_id_idx():
    """Test the deser_and_set_exp_id_idx validator."""
    specs = [
        SimpleSpec(experiment_id="original", sort_index=i, param_a=i).model_dump()
        for i in range(3)
    ]
    branch_spec = BranchesSpec(experiment_id="test", specs=specs)
    assert all(spec.experiment_id == "test" for spec in branch_spec.specs)
    assert all(spec.sort_index == i for i, spec in enumerate(branch_spec.specs))
