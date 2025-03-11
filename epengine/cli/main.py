"""CLI commands for epengine."""

from pathlib import Path
from typing import Literal

import click
import pandas as pd
from pydantic import BaseModel

from epengine.gis.submit import GisJobArgs
from epengine.models.leafs import AvailableWorkflowSpecs, WorkflowName


# TODO: move this
class Manifest(BaseModel):
    """A manifest for a sequence of jobs."""

    Name: str
    Jobs: list[GisJobArgs]


@click.group()
def cli():
    """CLI commands for epengine."""


@cli.group()
def submit():
    """Commands for job submission."""


@submit.group()
def gis():
    """Commands for GIS job submission."""


@gis.command()
@click.option(
    "--path",
    type=click.Path(exists=True),
    help="The path to the manifest file which will be used to schedule simulations.",
    prompt="Manifest file path (.yml)",
)
def manifest(path: Path):
    """Submit a GIS job."""
    import yaml

    from epengine.gis.submit import submit_gis_job

    with open(path) as f:
        manifest = yaml.safe_load(f)

    config = Manifest.model_validate(manifest)
    gis_jobs = [job for job in config.Jobs if isinstance(job, GisJobArgs)]
    click.echo(f"Submitting {len(gis_jobs)} GIS jobs.")
    for job in gis_jobs:
        submit_gis_job(config=job, log_fn=click.echo)


@gis.command()
@click.option(
    "--path",
    type=click.Path(exists=False),
    help="The path to the manifest file which will be used to schedule simulations.",
    prompt="Manifest file path (.yml)",
    default="manifest.yml",
)
@click.option(
    "--name",
    help="The name of the manifest.",
    prompt="Manifest name",
    default="GIS Job Submission",
)
def make(path: Path, name: str):
    """Make a manifest file."""
    import yaml

    job = GisJobArgs(
        gis_file="your-gis-data.geojson",
        db_file="your-db.db",
        component_map="your-component-map.yml",
        semantic_fields="your-semantic-fields.yml",
        experiment_id="your-experiment-id",
        cart_crs="EPSG:2285",
        leaf_workflow="simple",
    )
    manifest = Manifest(Name=name, Jobs=[job])
    with open(path, "w") as f:
        yaml.dump(manifest.model_dump(), f)
    click.echo(yaml.safe_dump(manifest.model_dump(), indent=2))
    click.echo(f"Manifest file created at {path}")


@gis.command()
@click.option(
    "--gis",
    type=click.Path(exists=True),
    help="The path to the GIS file which will be used to schedule simulations.",
    prompt="GIS file path (.geojson)",
)
@click.option(
    "--cart-crs",
    help="The crs of the cartesian coordinate system to project to.",
    prompt="Cartesian crs (e.g. EPSG:2285)",
)
@click.option(
    "--db",
    type=click.Path(exists=True),
    help="The path to the db file which will store components used to construct energy models.",
    prompt="DB file path (.db)",
)
@click.option(
    "--component-map",
    type=click.Path(exists=True),
    help="The path to the component map file which will be used to assign components to models according to GIS records.",
    prompt="Component map file path (.yaml)",
)
@click.option(
    "--semantic-fields",
    type=click.Path(exists=True),
    help="The path to the semantic fields file which will be used to assign semantic fields to models according to GIS records.",
    prompt="Semantic fields file path (.yaml)",
)
@click.option(
    "--leaf-workflow",
    help="The workflow to use.",
    prompt=f"Enter the workflow to use. [{'/'.join(AvailableWorkflowSpecs.keys())}]",
    type=click.Choice(list(AvailableWorkflowSpecs.keys())),
)
@click.option(
    "--experiment-id",
    help="The id of the experiment.",
    prompt="Experiment id in s3 storage",
)
@click.option(
    "--bucket",
    help="The bucket to use.",
    prompt="S3 Bucket",
    default="ml-for-bem",
)
@click.option(
    "--bucket-prefix",
    help="The prefix of the bucket to use.",
    prompt="Enter the prefix of the bucket to use.",
    default="hatchet",
)
@click.option(
    "--existing-artifacts",
    help="How to handle what happens when the experiment already exists in s3.",
    prompt="Enter how to handle what happens when the experiment already exists in s3. `forbid` will result in an error being thrown.",
    default="overwrite",
    type=click.Choice(["overwrite", "forbid"]),
)
@click.option(
    "--recursion-factor",
    help="The recursion factor for scatter/gather subdivision.",
    prompt="Enter the recursion factor for scatter/gather subdivision; using 10 will result in 10 subdivisions per level, with each responsible for n/10 tasks.  Recursive subdivision will continue until either the number of tasks per worker is fewer than the subdivision level or the max depth is achieved.",
    default=10,
)
@click.option(
    "--max-depth",
    help="The max depth for scatter/gather subdivision.",
    prompt="Enter the max depth for scatter/gather subdivision.  Recursive subdivision will continue until either the number of tasks per worker is fewer than the subdivision level or the max depth is achieved.",
    default=2,
)
def artifacts(
    gis: Path,
    cart_crs: str,
    db: Path,
    component_map: Path,
    semantic_fields: Path,
    leaf_workflow: WorkflowName,
    experiment_id: str,
    bucket: str,
    bucket_prefix: str,
    existing_artifacts: Literal["overwrite", "forbid"],
    recursion_factor: int,
    max_depth: int,
):
    """Submit a GIS job.

    Args:
        gis (Path): The path to the GIS file.
        cart_crs (str): The crs of the cartesian coordinate system to project to.
        db (Path): The path to the db file.
        component_map (Path): The path to the component map file.
        semantic_fields (Path): The path to the semantic fields file.
        leaf_workflow (WorkflowName): The workflow to use.
        experiment_id (str): The id of the experiment.
        bucket (str): The bucket to use.
        bucket_prefix (str): The prefix of the bucket to use.
        existing_artifacts (Literal["overwrite", "forbid"]): How to handle what happens when the experiment already exists in s3.
        recursion_factor (int): The recursion factor for scatter/gather subdivision.
        max_depth (int): The max depth for scatter/gather subdivision.
    """
    from epengine.gis.submit import GisJobArgs, submit_gis_job

    config = GisJobArgs(
        gis_file=str(gis),
        db_file=str(db),
        component_map=str(component_map),
        semantic_fields=str(semantic_fields),
        experiment_id=experiment_id,
        cart_crs=cart_crs,
        leaf_workflow=leaf_workflow,
        bucket=bucket,
        bucket_prefix=bucket_prefix,
        existing_artifacts=existing_artifacts,
        recursion_factor=recursion_factor,
        max_depth=max_depth,
    )

    click.echo("Submitting GIS job...")
    _result = submit_gis_job(
        config=config,
        log_fn=click.echo,
    )


@submit.command()
def idf():
    """Submit an IDF job."""
    click.echo("IDF job submission placeholder")


@cli.group()
def status():
    """Commands for job status."""


@status.command()
@click.option(
    "--workflow-run-id",
    help="The id of the workflow run.",
    prompt="Workflow run id",
)
@click.option(
    "--output-path",
    help="The path to the output file.",
    prompt="Output path",
)
def get(workflow_run_id: str, output_path: Path):
    """Get the results of a workflow run."""
    import asyncio

    from hatchet_sdk.client import new_client

    from epengine.utils.filesys import fetch_uri

    # TODO: add type for workflow_run_id

    if output_path.exists():
        click.echo(f"Output file already exists at {output_path}")
        return

    client = new_client()

    workflow = client.admin.get_workflow_run(workflow_run_id)
    res = asyncio.run(workflow.result())
    if "collect_children" in res:
        data = res["collect_children"]
        if "uri" in data:
            fetch_uri(data["uri"], output_path, use_cache=False)
        else:
            for key, df_dict in data.items():
                df = pd.DataFrame.from_dict(df_dict, orient="tight")
                df.to_hdf(output_path, key=key, mode="a")
        click.echo(f"Results saved to {output_path}")
    else:
        # TODO: check what happens if workflow does not exist.
        click.echo(f"Workflow run {workflow_run_id} has not completed yet.")
