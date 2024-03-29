"""Console script for ccres_disdrometer_processing."""
import logging
import sys
from pathlib import Path

import click
import toml

import ccres_disdrometer_processing.cli.preprocess_cli as preprocess_cli
from ccres_disdrometer_processing.__init__ import __version__
from ccres_disdrometer_processing.logger import LogLevels, init_logger
from ccres_disdrometer_processing.plot import plot, utils

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


@cli.command()
@click.argument(
    "file",
    type=click.Path(
        exists=True, dir_okay=False, file_okay=True, readable=True, resolve_path=True
    ),
)
@click.argument(
    "output_ql_overview",
    type=click.Path(
        exists=False, dir_okay=False, file_okay=True, writable=True, resolve_path=True
    ),
)
@click.argument(
    "output_ql_overview_zh",
    type=click.Path(
        exists=False, dir_okay=False, file_okay=True, writable=True, resolve_path=True
    ),
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
def preprocess_ql(file, output_ql_overview, output_ql_overview_zh, config_file):
    """Create quicklooks from preprocess netCDF files."""
    # 1 - check config and import configuration file if ok
    config = toml.load(config_file)

    # 2 - get preprocessed data
    data = utils.read_nc(file)
    # get date from data datetime
    date = data.time.to_dataframe().index.date[0]

    # 3 - Plot
    if data["weather_data_avail"].astype(bool) and ("ta" in data.data_vars):
        plot.plot_ql_overview(data, date, output_ql_overview, config, __version__)
    else:
        plot.plot_ql_overview_downgraded_mode(
            data, date, output_ql_overview, config, __version__
        )
    plot.plot_ql_overview_zh(data, date, output_ql_overview_zh, config, __version__)

    sys.exit(0)


if __name__ == "__main__":
    cli()
