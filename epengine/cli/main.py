"""CLI commands for epengine."""

from pathlib import Path
from typing import Literal

import click

from epengine.models.leafs import AvailableWorkflowSpecs, WorkflowName


@click.group()
def cli():
    """CLI commands for epengine."""


@cli.group()
def submit():
    """Commands for job submission."""


@submit.command()
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
def gis(
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
    from epengine.gis.submit import submit_gis_job

    click.echo("Submitting GIS job...")
    _result = submit_gis_job(
        gis_file=gis,
        db_file=db,
        component_map=component_map,
        semantic_fields=semantic_fields,
        experiment_id=experiment_id,
        cart_crs=cart_crs,
        leaf_workflow=leaf_workflow,
        bucket=bucket,
        bucket_prefix=bucket_prefix,
        existing_artifacts=existing_artifacts,
        recursion_factor=recursion_factor,
        max_depth=max_depth,
        log_fn=click.echo,
    )


@submit.command()
def idf():
    """Submit an IDF job."""
    click.echo("IDF job submission placeholder")


@cli.command()
def status():
    """Check the status of jobs."""
    click.echo("Job status placeholder")
