"""Console script for ccres_disdrometer_processing."""

import logging
import sys
from pathlib import Path

import click
import toml

import ccres_disdrometer_processing.cli.preprocess_cli as preprocess_cli
import ccres_disdrometer_processing.processing.preprocessed_file2processed as processing
from ccres_disdrometer_processing.__init__ import __version__
from ccres_disdrometer_processing.logger import LogLevels, init_logger
from ccres_disdrometer_processing.plot import plot, utils
from ccres_disdrometer_processing.utils import format_ql_file_prefix

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
        plot.plot_preprocessed_ql_overview(
            data, date, output_ql_overview, config, __version__
        )
    else:
        plot.plot_preprocessed_ql_overview_downgraded_mode(
            data, date, output_ql_overview, config, __version__
        )
    plot.plot_preprocessed_ql_overview_zh(
        data, date, output_ql_overview_zh, config, __version__
    )

    sys.exit(0)


@cli.command()
@click.option(
    "--yesterday",
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
    "--today",
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
    "--tomorrow",
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
def process(
    yesterday,
    today,
    tomorrow,
    config_file,
    output_file,
    verbosity,
):
    processing.process(yesterday, today, tomorrow, config_file, output_file, verbosity)
    click.echo("Processing : SUCCESS")
    sys.exit(0)


@cli.command()
@click.argument(
    "process-file",
    type=click.Path(
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    "--preprocess-yesterday",
    type=click.Path(
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    default=None,
)
@click.option(
    "--preprocess-today",
    type=click.Path(
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    required=True,
)
@click.option(
    "--preprocess-tomorrow",
    type=click.Path(
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    default=None,
)
@click.option(
    "--prefix-output-ql-summary",
    type=str,
    required=True,
    help="prefix need to be of form 'ql_summary'",
)
@click.option(
    "--prefix-output-ql-detailled",
    type=str,
    required=True,
    help="prefix need to be of form 'ql_detailled",
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
def process_ql(
    process_file,
    preprocess_yesterday,
    preprocess_today,
    preprocess_tomorrow,
    prefix_output_ql_summary,
    prefix_output_ql_detailled,
    config_file,
):
    """Create quicklooks from process netCDF files."""
    # create list of input preprocess files
    preprocess_files = [preprocess_yesterday, preprocess_today, preprocess_tomorrow]
    preprocess_files = [f for f in preprocess_files if f is not None]

    # check and format prefix
    prefix_output_ql_summary = format_ql_file_prefix(prefix_output_ql_summary)
    prefix_output_ql_detailled = format_ql_file_prefix(prefix_output_ql_detailled)

    # 1 - check config and import configuration file if ok
    config = toml.load(config_file)

    # 2a - get processed data
    ds_pro = utils.read_nc(process_file)

    # 2b - get preprocessed data
    if ds_pro.events.size != 0:
        ds_prepro = utils.read_and_concatenante_preprocessed_ds(
            ds_pro, preprocess_files
        )

        # 3 - Plot
        plot.plot_processed_ql_summary(
            ds_pro, prefix_output_ql_summary, config, __version__
        )
        plot.plot_processed_ql_detailled(
            ds_pro, ds_prepro, prefix_output_ql_detailled, config, __version__
        )

    sys.exit(0)


if __name__ == "__main__":
    cli()
