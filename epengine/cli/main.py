"""CLI commands for epengine."""

import click


@click.group()
def cli():
    """CLI commands for epengine."""


@cli.group()
def submit():
    """Commands for job submission."""


@submit.command()
def gis():
    """Submit a GIS job."""
    click.echo("GIS job submission placeholder")


@submit.command()
def idf():
    """Submit an IDF job."""
    click.echo("IDF job submission placeholder")


@cli.command()
def status():
    """Check the status of jobs."""
    click.echo("Job status placeholder")
