import logging
import tempfile
from pathlib import Path

import boto3
import pandas as pd
from archetypal import IDF
from archetypal.idfclass.sql import Sql
from hatchet_sdk import Hatchet
from hatchet_sdk.context import Context
from pydantic import AnyUrl, BaseModel

hatchet = Hatchet()
logger = logging.getLogger(__name__)

s3 = boto3.client("s3")


class SimulationSpec(BaseModel, arbitrary_types_allowed=True):
    idf_uri: AnyUrl
    epw_uri: AnyUrl
    experiment_id: str
    context: Context | None

    def local_path(self, pth: AnyUrl):
        path = pth.path
        return Path("/local_artifacts") / self.experiment_id / path

    @property
    def local_idf_path(self):
        return self.local_path(self.idf_uri)

    @property
    def local_epw_path(self):
        return self.local_path(self.epw_uri)

    def fetch_uri(self, uri: AnyUrl):
        local_path = self.local_path(uri)
        if uri.scheme == "s3":
            bucket = uri.host
            path = uri.path[1:]
            if not local_path.exists():
                self.log(f"Downloading {uri}...")
                local_path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, path, str(local_path))
            else:
                self.log(f"File {local_path} already exists, skipping download.")
        return local_path

    def fetch_idf(self):
        return self.fetch_uri(self.idf_uri)

    def fetch_epw(self):
        return self.fetch_uri(self.epw_uri)

    def log(self, msg: str):
        if self.context:
            self.context.log(msg)
        else:
            logger.info(msg)


@hatchet.workflow(
    name="download_and_simulate",
    on_events=["simulation:run"],
    timeout="4m",
    version="0.1",
)
class MyWorkflow:
    @hatchet.step()
    def step1(self, context: Context):
        data = context.workflow_input()
        spec = SimulationSpec(**data, context=context)
        local_epw_path = spec.fetch_epw()
        local_idf_path = spec.fetch_idf()

        with tempfile.TemporaryDirectory() as tmpdir:
            idf = IDF(local_idf_path, epw=local_epw_path, output_directory=tmpdir)
            context.log(f"Simulating {local_idf_path}...")
            idf.simulate()
            sql = Sql(idf.sql_file)
            dfs = postprocess(
                sql,
                data,
                tabular_lookups=[("AnnualBuildingUtilityPerformanceSummary", "End Uses")],
            )
        dfs = {k: v.to_dict(orient="tight") for k, v in dfs.items()}

        return dfs


def postprocess(
    sql: Sql,
    data: dict,
    tabular_lookups: list[tuple[str, str]],
) -> dict[str, pd.DataFrame]:
    dfs = {}
    for tabular_lookup in tabular_lookups:
        try:
            df = sql.tabular_data_by_name(*tabular_lookup)
        except Exception:
            logger.exception(f"Error while loading tabular data: {tabular_lookup}.")
            continue
        else:
            df = df.unstack()

            df = pd.DataFrame(df).T
            df.index = pd.MultiIndex.from_tuples([tuple(data.values())], names=list(data.keys()))
            df = df.dropna(axis=1, how="all")
            dfs["_".join(tabular_lookup).replace(" ", "_")] = df
    return dfs
