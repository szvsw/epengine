"""CLI commands for epengine."""

from pathlib import Path
from typing import Literal, cast

import click
import pandas as pd

from epengine.gis.models import GisJobArgs
from epengine.models.leafs import AvailableWorkflowSpecs, WorkflowName
from epengine.models.manifests import Manifest


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
    from epengine.gis.submit import submit_gis_job

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
    type=click.UUID,
)
@click.option(
    "--output-path",
    help="The path to the output file.",
    prompt="Output path",
    type=click.Path(exists=False),
)
def get(workflow_run_id: str, output_path: Path | str):
    """Get the results of a workflow run."""
    from hatchet_sdk.v0.client import new_client

    from epengine.utils.filesys import fetch_uri

    output_path = Path(output_path)
    if output_path.exists():
        click.echo(f"Output file already exists at {output_path}")
        return

    client = new_client()

    click.echo(f"Getting results for workflow run {workflow_run_id}...")
    workflow = client.admin.get_workflow_run(str(workflow_run_id))
    res = workflow.sync_result()
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


@cli.group()
def simulate():
    """Commands for simulating energy models."""


@simulate.command()
@click.option(
    "--db-path",
    help="The path to the db file.",
    prompt="DB file path",
    type=click.Path(exists=True),
)
@click.option(
    "--semantic-fields-path",
    help="The path to the semantic fields file.",
    prompt="Semantic fields file path",
    type=click.Path(exists=True),
)
@click.option(
    "--component-map-path",
    help="The path to the component map file.",
    type=click.Path(exists=True),
    prompt="Component map file path",
)
@click.option(
    "--latitude",
    help="The latitude of the site.",
    prompt="Latitude",
    type=click.FloatRange(min=-90, max=90),
)
@click.option(
    "--longitude",
    help="The longitude of the site.",
    prompt="Longitude",
    type=click.FloatRange(min=-180, max=180),
)
@click.option(
    "--wwr",
    help="The window-to-wall ratio of the model.",
    prompt="Window-to-wall ratio",
    type=click.FloatRange(min=0, max=1),
)
@click.option(
    "--num-floors",
    help="The number of floors in the model.",
    prompt="Number of floors",
    type=click.IntRange(min=1),
)
@click.option(
    "--f2f-height",
    help="The height of the floor to floor in the model.",
    prompt="Floor to floor height (min 2.5m)",
    type=click.FloatRange(min=2.5),
)
@click.option(
    "--exposed-basement-frac",
    help="The fraction of the basement that is exposed to the outside.",
    prompt="Exposed basement fraction",
    type=click.FloatRange(min=0, max=1),
)
@click.option(
    "--basement",
    help="The basement status of the model.",
    prompt="Basement",
    type=click.Choice([
        "none",
        "unoccupied_unconditioned",
        "unoccupied_conditioned",
        "occupied_unconditioned",
        "occupied_conditioned",
    ]),
)
@click.option(
    "--attic",
    help="The attic status of the model.",
    prompt="Attic",
    type=click.Choice([
        "none",
        "unoccupied_unconditioned",
        "unoccupied_conditioned",
        "occupied_unconditioned",
        "occupied_conditioned",
    ]),
)
def sbembox(  # noqa: C901
    db_path: Path,
    semantic_fields_path: Path,
    component_map_path: Path,
    latitude: float,
    longitude: float,
    wwr: float,
    num_floors: int,
    f2f_height: float,
    exposed_basement_frac: float,
    basement: Literal[
        "none",
        "unoccupied_unconditioned",
        "unoccupied_conditioned",
        "occupied_unconditioned",
        "occupied_conditioned",
    ],
    attic: Literal[
        "none",
        "unoccupied_unconditioned",
        "unoccupied_conditioned",
        "occupied_unconditioned",
        "occupied_conditioned",
    ],
):
    """Simulate a simple SBEM model.

    \b
    Assumptions:
    - The model footprint is 15m x 15m with faces perpendicular to the cardinal directions.
    - If an attic is not 'none', it's pitch is randomly set between 4/12 and 6/12 if
      it is unconditioned and unoccupied, otherwise it is set between 6/12 and 9/12
    - If an attic or basement is occupied, it's use fraction is randomly set between
      0.2 and 0.6 of the regular living space use fractions.
    - The EUI normalizing factor is always the total *conditioned* floor area.
    """  # noqa: D301
    import datetime

    import geopandas as gpd
    import yaml
    from epinterface.sbem.fields.spec import (
        CategoricalFieldSpec,
        NumericFieldSpec,
        SemanticModelFields,
    )
    from pydantic import AnyUrl
    from shapely.geometry import Point

    from epengine.gis.data.epw_metadata import closest_epw
    from epengine.models.shoebox_sbem import SBEMSimulationSpec

    query_pts = (
        gpd.GeoSeries([Point(longitude, latitude)])
        .set_crs("EPSG:4326")
        .to_crs("EPSG:3857")
    )
    epw = closest_epw(
        query_pts,
        source_filter="source in ['tmyx']",
        crs="EPSG:3857",
        log_fn=click.echo,
    )
    epw_path = epw.iloc[0]["path"]
    epw_name = epw.iloc[0]["name"]
    epw_ob_path = f"https://climate.onebuilding.org/{epw_path}"
    epw_uri = AnyUrl(epw_ob_path)
    click.echo(f"EPW: {epw_name}")

    with open(semantic_fields_path) as f:
        semantic_fields = yaml.safe_load(f)
    semantic_fields = SemanticModelFields.model_validate(semantic_fields)
    click.echo(f"Loaded semantic field model file from {semantic_fields_path}.")
    click.echo(f"Model name: {semantic_fields.Name}")

    def handle_categorical_field(
        field: CategoricalFieldSpec, input_mode: Literal["string", "int"]
    ):
        options = field.Options
        name = field.Name
        if len(options) == 1:
            value = options[0]
            click.echo(f"{name}: {value} (only one option allowed)")
            return name, value
        else:
            choice = (
                click.Choice(options)
                if input_mode == "string"
                else click.IntRange(0, len(options) - 1)
            )
            if input_mode == "int":
                for opt in range(len(options)):
                    click.echo(f"{opt}: {options[opt]}")
            value = click.prompt(f"{name}", type=choice)
            if input_mode == "int":
                value = options[value]
                click.echo(f"Selected: {value}\n")

            return name, value

    def handle_numeric_field(field: NumericFieldSpec):
        name = field.Name
        low, high = field.Min, field.Max
        value = -999999999
        while value < low or value > high:
            value = click.prompt("Enter the value", type=float)
            if value < low or value > high:
                click.echo(f"Value must be between {low} and {high}")
        return name, value

    click.echo("---")
    click.echo("Select the semantic field values for the model:")
    field_values = {}
    for field in semantic_fields.Fields:
        if isinstance(field, CategoricalFieldSpec):
            name, value = handle_categorical_field(field, input_mode="int")
            field_values[name] = value

        elif isinstance(field, NumericFieldSpec):
            name, value = handle_numeric_field(field)
            field_values[name] = value
        else:
            msg = f"Unsupported field type: {type(field)}"
            raise TypeError(msg)

    click.echo("---")
    click.echo("Semantic field values:")
    click.echo(yaml.safe_dump(field_values, indent=2, sort_keys=False))
    click.echo("---")

    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{current_timestamp}-local-test"
    spec = SBEMSimulationSpec(
        experiment_id=experiment_id,
        sort_index=0,
        db_uri=AnyUrl(f"file://{Path(db_path).absolute().as_posix()}"),
        semantic_fields_uri=AnyUrl(
            f"file://{Path(semantic_fields_path).absolute().as_posix()}"
        ),
        component_map_uri=AnyUrl(
            f"file://{Path(component_map_path).absolute().as_posix()}"
        ),
        semantic_field_context=field_values,
        neighbor_polys=[],
        neighbor_heights=[],
        neighbor_floors=[],
        rotated_rectangle="POLYGON ((0 0, 0 15, 15 15, 15 0, 0 0))",
        long_edge_angle=0,
        short_edge=15,
        long_edge=15,
        aspect_ratio=1,
        rotated_rectangle_area_ratio=1,
        wwr=wwr,
        height=num_floors * f2f_height,
        num_floors=num_floors,
        f2f_height=f2f_height,
        epwzip_uri=epw_uri,
        exposed_basement_frac=exposed_basement_frac,
        basement=basement,
        attic=attic,
    )
    idf, results, err_text = spec.run(log_fn=click.echo)

    # save?
    should_save = click.confirm("Save the IDF file?", default=False)
    if should_save:
        output_path = click.prompt("Output path", type=click.Path(exists=False))
        idf.saveas(output_path)
        click.echo(f"IDF file saved to {output_path}")

    click.echo("---")
    annual_results = (
        cast(pd.DataFrame, results["Energy"])
        .reset_index(drop=True)
        .iloc[0]
        .groupby(["Aggregation", "Meter"])
        .sum()
        .loc[["End Uses", "Utilities"]]
    )
    click.echo("Annual results:")
    click.echo(annual_results)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    monthly_end_uses = results.iloc[0]["Energy"]["End Uses"].unstack(level="Meter")
    annual_results["End Uses"].plot(kind="bar", ax=ax[0])
    monthly_end_uses.plot(kind="bar", ax=ax[1])
    ax[0].set_title("Annual End Uses")
    ax[1].set_title("Monthly End Uses")
    ax[0].set_ylabel("Energy (kWh/m²)")
    ax[1].set_ylabel("Energy (kWh/m²)")
    ax[0].set_xlabel("End Uses")
    ax[1].set_xlabel("Month")
    fig.suptitle(f"EUI: {annual_results['End Uses'].sum():.1f} kWh/m²")
    fig.tight_layout()
    plt.show()
