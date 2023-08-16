"""Console script for ccres_disdrometer_processing."""

import sys
from pathlib import Path

import click
import toml
import xarray as xr

sys.path.append(str(Path(__file__).parent.parent))

import open_disdro_netcdf as disdro
import open_weather_netcdf as weather
import scattering as scattering
from constants import BEAM_ORIENTATION, F_PARSIVEL, FREQ, E
from logger import LogLevels, init_logger

DISDRO_TYPES = ["parsivel-cloudnet"]
WS_TYPES = ["sirta-cloudnet"]
RADAR_TYPES = ["basta-cloudnet"]  # sera vu plus tard


@click.group()
@click.option("-v", "verbosity", count=True)
def cli(verbosity):
    log_level = LogLevels.get_by_verbosity_count(verbosity)
    init_logger(log_level)


@cli.command()
def status():
    print("Good, thanks !")


@cli.command()
@click.option("--disdro-type", type=click.Choice(DISDRO_TYPES), required=True)
@click.option("--disdro-file", type=click.Path(exists=True), required=True)
@click.option("--ws-type", type=click.Choice(WS_TYPES), required=True)
@click.option("--ws-file", type=click.Path(exists=True), required=True)
@click.option("--config-file", type=click.Path(exists=True), required=True)
@click.argument("output-file", type=click.Path())
def preprocess(disdro_type, disdro_file, ws_type, ws_file, config_file, output_file):
    """Command line interface for ccres_disdrometer_processing."""
    click.echo("CCRES disdrometer preprocessing : test CLI")

    config = toml.load(config_file)

    axrMethod = config["methods"]["AXIS_RATIO_METHOD"]
    strMethod = config["methods"]["FALL_SPEED_METHOD"]
    mieMethod = config["methods"]["COMPUTE_MIE_METHOD"]  # pymiecoated OR pytmatrix
    normMethod = config["methods"]["NORMALIZATION_METHOD"]  # measurement OR model

    # read weather-station data
    # ---------------------------------------------------------------------------------
    if ws_type == "sirta-cloudnet":
        weather_xr = weather.read_weather_cloudnet(ws_file)

    # read and preprocess disdrometer data
    # ---------------------------------------------------------------------------------

    if disdro_type == "parsivel-cloudnet":
        disdro_xr = disdro.read_parsivel_cloudnet(disdro_file)
        scatt = scattering.scattering_prop(
            disdro_xr.size_classes[0:-5],
            BEAM_ORIENTATION,
            FREQ,
            E,
            axrMethod=axrMethod,
            mieMethod=mieMethod,
        )
        disdro_xr = disdro.reflectivity_model(
            disdro_xr,
            scatt,
            len(disdro_xr.size_classes[0:-5]),
            F_PARSIVEL,
            FREQ,
            strMethod=strMethod,
            mieMethod=mieMethod,
            normMethod=normMethod,
        )

    final_data = xr.merge([weather_xr, disdro_xr])

    final_data.to_netcdf(output_file)

    return
