"""This module contains functions to postprocess and serialize results."""

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import pandas as pd
from archetypal.idfclass.sql import Sql
from pydantic import BaseModel

from epengine.utils.filesys import fetch_uri

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = object

logger = logging.getLogger(__name__)


# TODO: should we use this client as a session?
# TODO: should it be a parameter?


def postprocess(
    sql: Sql,
    index_data: dict,
    tabular_lookups: list[tuple[str, str]],
    columns: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Postprocess tabular data from the SQL file.

    Requests a series of Energy Plus table lookups and return the data in a
    dictionary of dataframes with a single row; the provided index data
    is configured as the MultiIndex of the dataframe.

    Args:
        sql (Sql): The sql object to query
        index_data (dict): The index data to use
        tabular_lookups (list[tuple[str, str]]): The tabular data to query
        columns (list[str], optional): The columns to keep. Defaults to None.

    Returns:
        dict[str, pd.DataFrame]: A dictionary of dataframes
    """
    dfs = {}
    for tabular_lookup in tabular_lookups:
        try:
            df = sql.tabular_data_by_name(*tabular_lookup)
        except Exception:
            logger.exception(f"Error while loading tabular data: {tabular_lookup}.")
            continue
        else:
            df = df[columns] if columns else df
            df = df.unstack()

            df = pd.DataFrame(df).T
            df.index = pd.MultiIndex.from_tuples(
                [tuple(index_data.values())],
                names=list(index_data.keys()),
            )
            df = df.dropna(axis=1, how="all")
            dfs["_".join(tabular_lookup).replace(" ", "_")] = df
    return dfs


def collate_subdictionaries(
    results: list[dict[str, dict]],
) -> dict[str, pd.DataFrame]:
    """Collate subdictionaries into a single dictionary of dataframes.

    Note that this assumes the dictionaries are in the tight orientation
    and that the index keys are the same across all dictionaries.

    Args:
        results (list[dict[str, dict]]): A list of dictionaries of dataframes

    Returns:
        dict[str, pd.DataFrame]: A dictionary of dataframes
    """
    dfs: dict[str, list[pd.DataFrame]] = {}
    for result in results:
        for key, _df in result.items():
            df = pd.DataFrame.from_dict(_df, orient="tight")
            if key not in dfs:
                dfs[key] = []
            dfs[key].append(df)

    data: dict[str, pd.DataFrame] = {k: pd.concat(v) for k, v in dfs.items()}

    return data


def serialize_df_dict(dfs: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Serialize a dictionary of dataframes into a dictionary of dictionaries.

    Args:
        dfs (dict[str, pd.DataFrame]): A dictionary of dataframes

    Returns:
        dict[str, dict]: A dictionary of dictionaries
    """
    return {k: v.to_dict(orient="tight") for k, v in dfs.items()}


ZipDataContent = TypeVar("ZipDataContent", bound=BaseModel)
ResultDataContent = TypeVar("ResultDataContent")


def separate_errors_and_safe_sim_results(
    ids: list[str],
    zip_data: list[ZipDataContent],
    results: list[ResultDataContent | BaseException],
):
    """Separates errored workflows from safe simulation results.

    Args:
        ids (list[str]): The list of workflow IDs.
        zip_data (list[ZipDataContent]): The list of data to return with rows.
        results (list[ResultDataContent | BaseException]): The list of results to separate.

    Returns:
        tuple[list[tuple[str, ZipDataContent, ResultDataContent]], list[tuple[str, ZipDataContent, BaseException]]]: A tuple containing the safe results and errored workflows.
    """
    errored_workflows: list[tuple[str, ZipDataContent, BaseException]] = []
    safe_results: list[tuple[str, ZipDataContent, ResultDataContent]] = []
    for workflow_id, spec, result in zip(ids, zip_data, results, strict=True):
        if isinstance(result, BaseException):
            errored_workflows.append((workflow_id, spec, result))
        else:
            safe_results.append((workflow_id, spec, result))
    return safe_results, errored_workflows


def create_errored_and_missing_df(
    errored_workflows: list[tuple[str, ZipDataContent, BaseException]],
    missing_results: list[tuple[str, ZipDataContent]],
):
    """Creates a DataFrame of errored and missing results.

    Args:
        errored_workflows (list[tuple[str, ZipDataContent, BaseException]]): The list of errored workflows.
        missing_results (list[tuple[str, ZipDataContent]]): The list of missing results.

    Returns:
        pd.DataFrame: The DataFrame of errored and missing results.
    """
    error_dfs: list[pd.DataFrame] = []
    for child_workflow_run_id, spec, result in errored_workflows:
        index_data = spec.model_dump(mode="json")
        index_data.update({"workflow_run_id": child_workflow_run_id})
        index = pd.MultiIndex.from_tuples(
            [tuple(index_data.values())],
            names=list(index_data.keys()),
        )
        row_data = {"missing": [False], "errored": [str(result)]}
        df = pd.DataFrame(row_data, index=index)
        error_dfs.append(df)

    for workflow_id, spec in missing_results:
        index_data = spec.model_dump(mode="json")
        index_data.update({"workflow_run_id": workflow_id})
        index = pd.MultiIndex.from_tuples(
            [tuple(index_data.values())],
            names=list(index_data.keys()),
        )
        row_data = {"missing": [True], "errored": [None]}
        df = pd.DataFrame(row_data, index=index)
        error_dfs.append(df)
    error_df = (
        pd.concat(error_dfs)
        if error_dfs
        else pd.DataFrame({"missing": [], "errored": []})
    )
    return error_df


def save_and_upload_results(
    collected_dfs: dict[str, pd.DataFrame],
    s3: S3Client,
    bucket: str,
    output_key: str,
    save_errors: bool = False,
):
    """Saves and uploads the collected dataframes to S3.

    Args:
        collected_dfs (dict[str, pd.DataFrame]): A dictionary containing the collected dataframes.
        bucket (str): The S3 bucket to upload the results to.
        output_key (str): The key to use for the uploaded results.
        save_errors (bool): Whether to save errors in the results.
        s3 (S3Client): The S3 client to use for uploading the results.

    Returns:
        uri (str): The URI of the uploaded results.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        local_path = f"{tempdir}/results.h5"
        for key, df in collected_dfs.items():
            if "error" in key.lower() and not save_errors:
                continue
            df.to_hdf(local_path, key=key, mode="a")
        s3.upload_file(Bucket=bucket, Key=output_key, Filename=local_path)
        uri = f"s3://{bucket}/{output_key}"
        return uri


# TODO: consider using a list based method which
# concats all at once rather than repeated concats
def update_collected_with_df(
    collected_dfs: dict[str, pd.DataFrame],
    key: str,
    df: pd.DataFrame,
):
    """Updates the collected dataframes with a new DataFrame.

    Note that this function mutates the collected_dfs dictionary and does
    not return a value.

    Args:
        collected_dfs (dict[str, pd.DataFrame]): A dictionary containing the collected dataframes.
        key (str): The key to use for the new DataFrame.
        df (pd.DataFrame): The DataFrame to add to the collected dataframes.

    Returns:
        None (mutates the collected_dfs dictionary).
    """
    if key in collected_dfs:
        df = pd.concat([collected_dfs[key], df], axis=0)
    collected_dfs[key] = df


def handle_explicit_result(collected_dfs: dict[str, pd.DataFrame], result: dict):
    """Updates the collected dataframes with an explicit result.

    Note that this function mutates the collected_dfs dictionary and does
    not return a value.

    Args:
        collected_dfs (dict[str, pd.DataFrame]): A dictionary containing the collected dataframes.
        result (dict): The explicit result to handle.

    Returns:
        None (mutates the collected_dfs dictionary).
    """
    for key, df_dict in result.items():
        df = pd.DataFrame.from_dict(df_dict, orient="tight")
        update_collected_with_df(collected_dfs, key, df)
        del df, df_dict


def handle_referenced_result(collected_dfs: dict[str, pd.DataFrame], uri: str):
    """Fetches a result from a given URI and updates the collected dataframes.

    Note that this function mutates the collected_dfs dictionary and does
    not return a value.

    Args:
        collected_dfs (dict[str, pd.DataFrame]): A dictionary containing the collected dataframes.
        uri (str): The URI of the result to fetch.

    Returns:
        None (mutates the collected_dfs dictionary).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        res_path = fetch_uri(
            uri=uri,
            use_cache=False,
            local_path=Path(tmpdir) / "result.h5",
        )
        # list the keys in the h5 file
        with pd.HDFStore(res_path, mode="r") as store:
            keys = store.keys()
            for key in keys:
                # get the dataframes
                df = cast(pd.DataFrame, store[key])
                update_collected_with_df(collected_dfs, key, df)
                del df


def combine_recurse_results(results: list[dict[str, Any]]):
    """Combines the results of recursive operations into a single dictionary of DataFrames.

    The recursive returns may have been uris or explicit results. This function combines them into a single dictionary
    of DataFrames.

    Args:
      results: A list of dictionaries representing the results of recursive operations.

    Returns:
      collected_dfs: A dictionary containing the combined DataFrames, where the keys are the URIs of the DataFrames.

    Raises:
      CombineRecurseResultsMultipleKeysError: If a result dictionary contains more than one key when it has a "uri" key.
    """
    collected_dfs: dict[str, pd.DataFrame] = {}

    for result in results:
        if "uri" not in result:
            handle_explicit_result(collected_dfs, result)
        else:
            if len(result) > 1:
                raise CombineRecurseResultsMultipleKeysError
            uri = result["uri"]
            handle_referenced_result(collected_dfs, uri)
    return collected_dfs


class CombineRecurseResultsMultipleKeysError(ValueError):
    """Raised when a result dictionary contains more than one key when it has a "uri" key."""

    def __init__(self):
        """Initializes the error."""
        super().__init__(
            "Cannot have both a uri and other keys in the results dict when combining recurse results"
        )
