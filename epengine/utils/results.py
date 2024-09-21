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
            df.index = pd.MultiIndex.from_tuples([tuple(index_data.values())], names=list(index_data.keys()))
            df = df.dropna(axis=1, how="all")
            dfs["_".join(tabular_lookup).replace(" ", "_")] = df
    return dfs


def collate_subdictionaries(
    results: list[dict[str, dict]],
) -> dict[str, pd.DataFrame]:
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
    return {k: v.to_dict(orient="tight") for k, v in dfs.items()}
