"""Simulate an EnergyPlus ubem shoebox model with associated artifacts."""

import logging

import numpy as np
import pandas as pd
from hatchet_sdk.context import Context

from epengine.hatchet import hatchet
from epengine.models.leafs import SimpleSpec
from epengine.models.mixins import WithHContext
from epengine.utils.results import serialize_df_dict

logger = logging.getLogger(__name__)


class SimpleSpecWithContext(WithHContext, SimpleSpec):
    """A simple workflow specification with a Hatchet Context."""

    pass

    def run(self):
        """Run the simulation."""
        self.log(f"Param a says: {self.param_a}")
        return float(np.random.uniform(0, 1))


@hatchet.workflow(
    name="simple",
    timeout="10m",
    version="0.3",
    schedule_timeout="1000m",
)
class SimpleTest:
    """A workflow to simulate an EnergyPlus model."""

    @hatchet.step(name="simulate", timeout="10m", retries=2)
    def simulate(self, context: Context):
        """Simulate an EnergyPlus Shoebox UBEM model.

        Args:
            context (Context): The context of the workflow

        Returns:
            dict: A dictionary of dataframes with results.
        """
        data = context.workflow_input()
        data["hcontext"] = context
        spec = SimpleSpecWithContext(**data)
        res = spec.run()
        results = pd.DataFrame(
            {"data": [res]}, index=pd.Index([spec.param_a], name="param_a")
        )
        results.columns.name = "data"
        dfs = {"results": results}

        dfs = serialize_df_dict(dfs)

        return dfs


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    import boto3
    from hatchet_sdk import new_client

    client = new_client()
    # spec = SimpleSpec(
    #     experiment_id="test",
    #     sort_index=0,
    #     param_a=2,
    # )
    # client.admin.run_workflow("simple_test", spec.model_dump(mode="json"))

    specs = [
        SimpleSpec(experiment_id="test", sort_index=i, param_a=i).model_dump()
        for i in range(100)
    ]
    specs = pd.DataFrame(specs)
    experiment_id = "DELETE-simple-test-2025-jan"

    s3 = boto3.client("s3")
    bucket = "ml-for-bem"
    specs_key = f"hatchet/{experiment_id}/specs.pq"

    with tempfile.TemporaryDirectory() as tempdir:
        specs_path = Path(tempdir) / "specs.pq"
        specs.to_parquet(specs_path, index=False)
        s3.upload_file(
            Filename=specs_path.as_posix(),
            Bucket=bucket,
            Key=specs_key,
        )
        s3_uri = f"s3://{bucket}/{specs_key}"

    # WARNING: currently, when max_depth = 2, with uri upload, it sends too many.
    workflow_payload = {
        "specs": s3_uri,
        "workflow_name": "simple",
        "recursion_map": {"factor": 3, "max_depth": 2},
        "experiment_id": experiment_id,
        "bucket": bucket,
    }

    client.admin.run_workflow("scatter_gather_recursive", workflow_payload)
