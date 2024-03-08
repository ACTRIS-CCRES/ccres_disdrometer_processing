"""Console script for ccres_disdrometer_processing."""
import datetime as dt
import logging
import sys
from pathlib import Path

import click

import ccres_disdrometer_processing.cli.preprocess_cli as preprocess_cli
from ccres_disdrometer_processing.__init__ import __version__
from ccres_disdrometer_processing.cli.plot import plot, utils
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


@cli.command()
@click.argument(
    "date",
    type=click.DateTime(formats=["%Y%m%d", "%Y-%m-%d"]),
    default=dt.datetime.now(),
)
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
@click.argument("config", default="ccres_disdrometer_processing/cli/conf/conf.py")
@click.argument("parameter", default="ccres_disdrometer_processing/cli/conf/params.py")
def preprocess_ql(
    date, file, output_ql_overview, output_ql_overview_zh, config, parameter
):
    """Create quicklooks from preprocess netCDF files."""
    # 1 - check config and import configuration file if ok
    conf = utils.load_module("conf", config)
    params = utils.load_module("parameters", parameter)

    # 2 - get preprocessed data
    data = utils.read_nc(file)

    # 3 - Plot
    if data.attrs["weather_data_avail"] and ("ta" in data.data_vars):
        plot.plot_ql_overview(data, date, output_ql_overview, conf, params, __version__)
    else:
        plot.plot_ql_overview_downgraded_mode(
            data, date, output_ql_overview, conf, params, __version__
        )
    plot.plot_ql_overview_zh(
        data, date, output_ql_overview_zh, conf, params, __version__
    )

    sys.exit(0)


if __name__ == "__main__":
    cli()
