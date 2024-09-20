import asyncio
import tempfile

import boto3
from hatchet_sdk import Context

from epengine.hatchet import hatchet
from epengine.models.configs import SimulationsSpec
from epengine.utils.results import collate_subdictionaries, serialize_df_dict

s3 = boto3.client("s3")


@hatchet.workflow(
    name="scatter_gather",
    version="0.2",
    timeout="1200m",
    on_events=["simulations:fanout"],
)
class Fanout:
    @hatchet.step(timeout="1200m")
    async def spawn_children(self, context: Context):
        workflow_input = context.workflow_input()
        specs = SimulationsSpec.from_payload(workflow_input)
        specs.hcontext = context

        promises = []
        ids = []

        for i, spec in enumerate(specs.specs):
            if i % 1000 == 0:
                context.log(f"Queing {i}th child workflow...")
            task = await context.aio.spawn_workflow(
                "simulate_epw_idf",
                spec.model_dump(mode="json"),
                options={
                    "additional_metadata": {
                        "index": i,  # pyright: ignore [reportArgumentType]
                    },
                },
            )
            # TODO: check error handling
            promise = task.result()
            promises.append(promise)
            ids.append(task.workflow_run_id)

        results = await asyncio.gather(*promises, return_exceptions=True)

        # errors = [
        #     (run_id, err)
        #     for run_id, err in zip(ids, results)
        #     if isinstance(err, Exception)
        # ]
        # for run_id, err in errors:
        #     context.log(f"Error in child workflow {run_id}: {err}")

        collated_dfs = collate_subdictionaries(results)

        if specs.bucket:
            # save as hdfs
            workflow_run_id = context.workflow_run_id()
            # TODO: hatchet prefix should come from task!
            output_key = f"hatchet/{specs.experiment_id}/results/{workflow_run_id}.h5"
            with tempfile.TemporaryDirectory() as tempdir:
                local_path = f"{tempdir}/results.h5"
                for key, df in collated_dfs.items():
                    df.to_hdf(local_path, key=key, mode="a")
                s3.upload_file(Bucket=specs.bucket, Key=output_key, Filename=local_path)
                uri = f"s3://{specs.bucket}/{output_key}"
                return {"uri": uri}
        else:
            dfs = serialize_df_dict(collated_dfs)
            return dfs
