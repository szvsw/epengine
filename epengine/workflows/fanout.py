import asyncio

from hatchet_sdk import Context

from epengine.hatchet import hatchet
from epengine.models.configs import SimulationsSpec
from epengine.utils.results import collate_subdictionaries, serialize_df_dict


@hatchet.workflow(
    name="scatter_gather",
    version="0.2",
    timeout="20m",
    on_events=["simulations:fanout"],
)
class Fanout:
    @hatchet.step(timeout="20m")
    async def spawn_children(self, context: Context):
        workflow_input = context.workflow_input()
        specs = SimulationsSpec(**workflow_input, hcontext=context)

        promises = []
        ids = []

        for i, spec in enumerate(specs.specs):
            # TODO: workflowname should be an enum probably, or dynamic on input
            # TODO: passing in child index - can meta be accessed in the child?
            # or alternatively, can spawn index?
            task = await context.aio.spawn_workflow(
                "simulate_epw_idf",
                spec.model_dump(mode="json"),
                options={"additional_metadata": {"index": i}},
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

        # TODO: will need to send large files to a storage service
        # > 4MB
        dfs = serialize_df_dict(collate_subdictionaries(results))

        return dfs
