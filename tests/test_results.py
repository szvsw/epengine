"""Tests for the results module."""

import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import BaseModel

from epengine.utils.results import (
    collate_subdictionaries,
    create_errored_and_missing_df,
    handle_explicit_result,
    save_and_upload_results,
    separate_errors_and_safe_sim_results,
    serialize_df_dict,
    update_collected_with_df,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def check_dict_of_df_equality(
    dict1: dict[str, pd.DataFrame], dict2: dict[str, pd.DataFrame]
) -> bool:
    """Check the equality of two dictionaries of DataFrames.

    Args:
        dict1 (dict[str, pd.DataFrame]): The first dictionary of DataFrames.
        dict2 (dict[str, pd.DataFrame]): The second dictionary of DataFrames.

    Returns:
        result (bool): True if the two dictionaries are equal, False otherwise.
    """
    if dict1.keys() != dict2.keys():
        return False
    return all(dict1[key].equals(dict2[key]) for key in dict1)


def test_collate_subdictionaries():
    """Test the collate_subdictionaries function."""
    # Test case 1: Single dictionary
    results = [
        {
            "key1": pd.DataFrame({"A": [1], "B": [2]}, index=[2]).to_dict(
                orient="tight"
            ),
            "key2": pd.DataFrame({"C": [3], "D": [4]}, index=[2]).to_dict(
                orient="tight"
            ),
        },
        {
            "key2": pd.DataFrame({"C": [5], "D": [6]}, index=[7]).to_dict(
                orient="tight"
            ),
        },
    ]
    expected_output = {
        "key1": pd.DataFrame({"A": [1], "B": [2]}, index=[2]),
        "key2": pd.DataFrame({"C": [3, 5], "D": [4, 6]}, index=[2, 7]),
    }
    assert check_dict_of_df_equality(collate_subdictionaries(results), expected_output)


def test_collate_subdictionaries_missing_keys():
    """Test the collate_subdictionaries function."""
    results = [
        {"key1": pd.DataFrame({"A": [1], "B": [2]}, index=[0]).to_dict(orient="tight")},
        {"key2": pd.DataFrame({"B": [3], "C": [4]}, index=[0]).to_dict(orient="tight")},
        {"key1": pd.DataFrame({"C": [5], "D": [6]}, index=[1]).to_dict(orient="tight")},
    ]
    expected_output = {
        "key1": pd.DataFrame(
            {
                "A": [1, None],
                "B": [2, None],
                "C": [None, 5],
                "D": [None, 6],
            },
            index=[0, 1],
        ),
        "key2": pd.DataFrame({"B": [3], "C": [4]}, index=[0]),
    }
    assert check_dict_of_df_equality(collate_subdictionaries(results), expected_output)


def test_collate_subdictionaries_empty_list():
    """Test the collate_subdictionaries function."""
    results = []
    expected_output = {}
    assert check_dict_of_df_equality(collate_subdictionaries(results), expected_output)


def test_collate_subdictionaries_non_overlapping_keys():
    """Test the collate_subdictionaries function."""
    results = [
        {"key1": pd.DataFrame({"A": [1], "B": [2]}, index=[0]).to_dict(orient="tight")},
        {"key2": pd.DataFrame({"C": [3], "D": [4]}, index=[0]).to_dict(orient="tight")},
        {"key3": pd.DataFrame({"E": [5], "F": [6]}, index=[0]).to_dict(orient="tight")},
    ]
    expected_output = {
        "key1": pd.DataFrame({"A": [1], "B": [2]}, index=[0]),
        "key2": pd.DataFrame({"C": [3], "D": [4]}, index=[0]),
        "key3": pd.DataFrame({"E": [5], "F": [6]}, index=[0]),
    }
    assert check_dict_of_df_equality(collate_subdictionaries(results), expected_output)


def test_serialize_df_dict():
    """Test the serialize_df_dict function."""
    # Test case 1: Single dictionary
    df_dict = {
        "key1": pd.DataFrame({"A": [1], "B": [2]}, index=[2]),
        "key2": pd.DataFrame({"C": [3, 5], "D": [4, 6]}, index=[2, 7]),
    }

    expected_output = {
        "key1": {
            "index": [2],
            "columns": ["A", "B"],
            "data": [[1, 2]],
            "index_names": [None],
            "column_names": [None],
        },
        "key2": {
            "index": [2, 7],
            "columns": ["C", "D"],
            "data": [[3, 4], [5, 6]],
            "index_names": [None],
            "column_names": [None],
        },
    }

    assert serialize_df_dict(df_dict) == expected_output


class TestZipDataContent(BaseModel):
    """Test class for separate_errors_and_safe_sim_results."""

    name: str
    value: int


def test_separate_errors_and_safe_sim_results_all_safe():
    """Test case where all results are safe (no exceptions)."""
    ids = ["id1", "id2", "id3"]
    zip_data = [
        TestZipDataContent(name="test1", value=10),
        TestZipDataContent(name="test2", value=20),
        TestZipDataContent(name="test3", value=30),
    ]
    results: list[str | BaseException] = [
        "result1",
        "result2",
        "result3",
    ]  # All safe results

    safe_results, errored_results = separate_errors_and_safe_sim_results(
        ids, zip_data, results
    )

    # Assert all results are safe
    assert len(safe_results) == 3
    assert len(errored_results) == 0

    # Check that the safe_results match expected values
    expected_safe_results = list(zip(ids, zip_data, results, strict=True))
    assert safe_results == expected_safe_results


def test_separate_errors_and_safe_sim_results_all_errors():
    """Test case where all results are exceptions."""
    ids = ["id1", "id2", "id3"]
    zip_data = [
        TestZipDataContent(name="test1", value=10),
        TestZipDataContent(name="test2", value=20),
        TestZipDataContent(name="test3", value=30),
    ]
    results = [
        BaseException("Error1"),
        BaseException("Error2"),
        BaseException("Error3"),
    ]

    safe_results, errored_results = separate_errors_and_safe_sim_results(
        ids, zip_data, results
    )

    # Assert all results are errored
    assert len(safe_results) == 0
    assert len(errored_results) == 3

    # Check that the errored_results match expected values
    expected_errored_results = list(zip(ids, zip_data, results, strict=False))
    assert errored_results == expected_errored_results


def test_separate_errors_and_safe_sim_results_mixed():
    """Test case with a mix of safe results and exceptions."""
    ids = ["id1", "id2", "id3", "id4"]
    zip_data = [
        TestZipDataContent(name="test1", value=10),
        TestZipDataContent(name="test2", value=20),
        TestZipDataContent(name="test3", value=30),
        TestZipDataContent(name="test4", value=40),
    ]
    results = ["result1", BaseException("Error2"), "result3", BaseException("Error4")]

    safe_results, errored_results = separate_errors_and_safe_sim_results(
        ids, zip_data, results
    )

    # Assert correct number of safe and errored results
    assert len(safe_results) == 2
    assert len(errored_results) == 2

    # Expected safe and errored results
    expected_safe_results = [
        (ids[0], zip_data[0], results[0]),
        (ids[2], zip_data[2], results[2]),
    ]
    expected_errored_results = [
        (ids[1], zip_data[1], results[1]),
        (ids[3], zip_data[3], results[3]),
    ]

    assert safe_results == expected_safe_results
    assert errored_results == expected_errored_results


def test_separate_errors_and_safe_sim_results_unequal_lengths():
    """Test case where input lists have unequal lengths."""
    ids = ["id1", "id2"]
    zip_data = [
        TestZipDataContent(name="test1", value=10),
        TestZipDataContent(name="test2", value=20),
        TestZipDataContent(name="test3", value=30),
    ]
    results: list[str | BaseException] = ["result1", "result2"]

    with pytest.raises(ValueError):
        separate_errors_and_safe_sim_results(ids, zip_data, results)


def test_separate_errors_and_safe_sim_results_custom_exception():
    """Test case using a custom exception class."""

    class CustomException(Exception):
        pass

    ids = ["id1"]
    zip_data = [TestZipDataContent(name="test1", value=10)]
    results: list[str | BaseException] = [CustomException("Custom error")]

    safe_results, errored_results = separate_errors_and_safe_sim_results(
        ids, zip_data, results
    )

    # Assert that the result is treated as an error
    assert len(safe_results) == 0
    assert len(errored_results) == 1

    expected_errored_results = [(ids[0], zip_data[0], results[0])]
    assert errored_results == expected_errored_results


def test_create_errored_and_missing_df_empty():
    """Test with empty errored_workflows and missing_results."""
    errored_workflows = []
    missing_results = []
    error_df = create_errored_and_missing_df(errored_workflows, missing_results)

    expected_df = pd.DataFrame({"missing": [], "errored": []})
    pd.testing.assert_frame_equal(error_df, expected_df)


def test_create_errored_and_missing_df_only_errored():
    """Test with only errored workflows."""
    errored_workflows = [
        (
            "workflow1",
            TestZipDataContent(name="value1", value=1),
            BaseException("Error1"),
        ),
        (
            "workflow2",
            TestZipDataContent(name="value2", value=2),
            BaseException("Error2"),
        ),
    ]
    missing_results = []
    error_df = create_errored_and_missing_df(errored_workflows, missing_results)

    records = []
    for child_workflow_run_id, spec, result in errored_workflows:
        index_data = spec.model_dump(mode="json")
        index_data.update({"workflow_run_id": child_workflow_run_id})
        row = {"missing": False, "errored": str(result)}
        index = pd.MultiIndex.from_tuples(
            [tuple(index_data.values())],
            names=list(index_data.keys()),
        )
        df = pd.DataFrame(row, index=index)
        records.append(df)

    expected_df = pd.concat(records)
    pd.testing.assert_frame_equal(error_df, expected_df)


def test_create_errored_and_missing_df_only_missing():
    """Test with only missing results."""
    errored_workflows = []
    missing_results = [
        ("workflow3", TestZipDataContent(name="value3", value=3)),
        ("workflow4", TestZipDataContent(name="value4", value=4)),
    ]
    error_df = create_errored_and_missing_df(errored_workflows, missing_results)

    records = []
    for workflow_id, spec in missing_results:
        index_data = spec.model_dump(mode="json")
        index_data.update({"workflow_run_id": workflow_id})
        row = {"missing": True, "errored": None}
        index = pd.MultiIndex.from_tuples(
            [tuple(index_data.values())],
            names=list(index_data.keys()),
        )
        df = pd.DataFrame(row, index=index)
        records.append(df)

    expected_df = pd.concat(records)
    pd.testing.assert_frame_equal(error_df, expected_df)


def test_create_errored_and_missing_df_mixed():
    """Test with both errored workflows and missing results."""
    errored_workflows = [
        (
            "workflow1",
            TestZipDataContent(name="value1", value=1),
            BaseException("Error1"),
        ),
    ]
    missing_results = [
        ("workflow2", TestZipDataContent(name="value2", value=2)),
    ]
    error_df = create_errored_and_missing_df(errored_workflows, missing_results)

    records = []

    # From errored_workflows
    for child_workflow_run_id, spec, result in errored_workflows:
        index_data = spec.model_dump(mode="json")
        index_data.update({"workflow_run_id": child_workflow_run_id})
        row = {"missing": False, "errored": str(result)}
        index = pd.MultiIndex.from_tuples(
            [tuple(index_data.values())],
            names=list(index_data.keys()),
        )
        df = pd.DataFrame(row, index=index)
        records.append(df)

    # From missing_results
    for workflow_id, spec in missing_results:
        index_data = spec.model_dump(mode="json")
        index_data.update({"workflow_run_id": workflow_id})
        row = {"missing": True, "errored": None}
        index = pd.MultiIndex.from_tuples(
            [tuple(index_data.values())],
            names=list(index_data.keys()),
        )
        df = pd.DataFrame(row, index=index)
        records.append(df)

    expected_df = pd.concat(records)
    pd.testing.assert_frame_equal(error_df.sort_index(), expected_df.sort_index())


def test_save_and_upload_results():
    """Test saving and uploading results with and without errors."""
    # Prepare test data
    collected_dfs = {
        "data1": pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
        "errors": pd.DataFrame({"error_msg": ["Error1", "Error2"]}),
        "Data_Error": pd.DataFrame({"error_msg": ["Error3"]}),
        "data2": pd.DataFrame({"C": [5, 6], "D": [7, 8]}),
    }
    bucket = "test-bucket"
    output_key = "test/output/results.h5"

    # Mock S3 client
    s3_client = MagicMock()

    # Test with save_errors=False
    with patch("pandas.DataFrame.to_hdf") as mock_to_hdf:
        uri = save_and_upload_results(
            collected_dfs=collected_dfs,
            s3=s3_client,
            bucket=bucket,
            output_key=output_key,
            save_errors=False,
        )

        # Check that DataFrames with 'error' in key are not saved
        assert mock_to_hdf.call_count == 2
        saved_keys = [call[1]["key"] for call in mock_to_hdf.call_args_list]
        assert "data1" in saved_keys
        assert "data2" in saved_keys
        assert "errors" not in saved_keys
        assert "Data_Error" not in saved_keys

        # Verify that s3.upload_file was called correctly
        s3_client.upload_file.assert_called_once()
        args, kwargs = s3_client.upload_file.call_args
        assert kwargs["Bucket"] == bucket
        assert kwargs["Key"] == output_key

        # Check the returned URI
        assert uri == f"s3://{bucket}/{output_key}"

    # Reset mocks
    s3_client.reset_mock()
    mock_to_hdf.reset_mock()

    # Test with save_errors=True
    with patch("pandas.DataFrame.to_hdf") as mock_to_hdf:
        uri = save_and_upload_results(
            collected_dfs=collected_dfs,
            s3=s3_client,
            bucket=bucket,
            output_key=output_key,
            save_errors=True,
        )

        # Check that all DataFrames are saved
        assert mock_to_hdf.call_count == 4
        saved_keys = [call[1]["key"] for call in mock_to_hdf.call_args_list]
        assert "data1" in saved_keys
        assert "data2" in saved_keys
        assert "errors" in saved_keys
        assert "Data_Error" in saved_keys

        # Verify that s3.upload_file was called correctly
        s3_client.upload_file.assert_called_once()
        args, kwargs = s3_client.upload_file.call_args
        assert kwargs["Bucket"] == bucket
        assert kwargs["Key"] == output_key

        # Check the returned URI
        assert uri == f"s3://{bucket}/{output_key}"


def test_update_collected_with_df_new_key():
    """Test updating collected_dfs with a new key."""
    collected_dfs = {}
    key = "test_key"
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    update_collected_with_df(collected_dfs, key, df)
    assert key in collected_dfs
    pd.testing.assert_frame_equal(collected_dfs[key], df)


def test_update_collected_with_df_existing_key():
    """Test updating collected_dfs with an existing key."""
    df_existing = pd.DataFrame({"A": [5, 6], "B": [7, 8]})
    collected_dfs = {"test_key": df_existing}
    key = "test_key"
    df_new = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    update_collected_with_df(collected_dfs, key, df_new)
    expected_df = pd.concat([df_existing, df_new], axis=0)
    pd.testing.assert_frame_equal(collected_dfs[key], expected_df)


def test_handle_explicit_result_new_keys():
    """Test handling explicit results with new keys."""
    collected_dfs = {}
    result = {
        "df1": pd.DataFrame({"A": [1, 3], "B": [2, 4]}, index=[0, 1]).to_dict(
            orient="tight"
        ),
        "df2": pd.DataFrame({"C": [5], "D": [6]}, index=[0]).to_dict(orient="tight"),
    }
    handle_explicit_result(collected_dfs, result)

    # Expected DataFrames
    expected_df1 = pd.DataFrame({"A": [1, 3], "B": [2, 4]}, index=[0, 1])
    expected_df2 = pd.DataFrame({"C": [5], "D": [6]}, index=[0])

    assert "df1" in collected_dfs
    assert "df2" in collected_dfs
    pd.testing.assert_frame_equal(collected_dfs["df1"], expected_df1)
    pd.testing.assert_frame_equal(collected_dfs["df2"], expected_df2)


def test_handle_explicit_result_existing_keys():
    """Test handling explicit results with existing keys."""
    df_existing = pd.DataFrame({"A": [10, 20]}, index=[0, 1])
    collected_dfs = {"df1": df_existing}
    result = {
        "df1": pd.DataFrame({"A": [30]}, index=[2]).to_dict(orient="tight"),
    }
    handle_explicit_result(collected_dfs, result)

    # Expected DataFrame after concatenation
    df_new = pd.DataFrame({"A": [30]}, index=[2])
    expected_df1 = pd.concat([df_existing, df_new], axis=0)
    pd.testing.assert_frame_equal(collected_dfs["df1"], expected_df1)


def test_handle_explicit_result_empty_result():
    """Test handling an empty explicit result."""
    collected_dfs = {}
    result = {}
    handle_explicit_result(collected_dfs, result)
    assert collected_dfs == {}


def test_handle_explicit_result_invalid_format():
    """Test handling explicit result with invalid format."""
    collected_dfs = {}
    result = {
        "df1": "invalid_data",
    }
    with pytest.raises(ValueError):
        handle_explicit_result(collected_dfs, result)
