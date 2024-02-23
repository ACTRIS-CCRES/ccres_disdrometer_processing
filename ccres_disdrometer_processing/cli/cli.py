"""Console script for ccres_disdrometer_processing."""

import logging
from pathlib import Path

import click

import ccres_disdrometer_processing.cli.preprocess_cli as preprocess_cli
from ccres_disdrometer_processing.logger import LogLevels, init_logger

lgr = logging.getLogger(__name__)


@click.group()
@click.option("-v", "verbosity", count=True)
def cli(verbosity):
    log_level = LogLevels.get_by_verbosity_count(verbosity)
    init_logger(log_level)


@cli.command()
@click.option("-v", "verbosity", count=True)
def status(verbosity):
    print(verbosity)
    print("Good, thanks !")


@cli.command()
@click.option(
    "--disdro-file",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
        resolve_path=True,
        readable=True,
    ),
    required=True,
)
@click.option(
    "--ws-file",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
        resolve_path=True,
        readable=True,
    ),
    required=False,
    default=None,
)
@click.option(
    "--radar-file",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
        resolve_path=True,
        readable=True,
    ),
    required=True,
)
@click.option(
    "--config-file",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
        resolve_path=True,
        readable=True,
    ),
    required=True,
)
@click.argument(
    "output-file",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path, resolve_path=True),
)
@click.option("-v", "verbosity", count=True)
def preprocess(disdro_file, ws_file, radar_file, config_file, output_file, verbosity):
    return preprocess_cli.preprocess(
        disdro_file, ws_file, radar_file, config_file, output_file, verbosity
    )


if __name__ == "__main__":
    cli()
