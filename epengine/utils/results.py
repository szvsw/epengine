"""This module contains functions to postprocess and serialize results."""

import logging

import pandas as pd
from archetypal.idfclass.sql import Sql

logger = logging.getLogger(__name__)


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
