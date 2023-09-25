"""Console script for ccres_disdrometer_processing."""

import sys
from pathlib import Path

import click
import toml
import xarray as xr

sys.path.append(str(Path(__file__).parent.parent))

import open_disdro_netcdf as disdro
import open_radar_netcdf as radar
import open_weather_netcdf as weather
import scattering as scattering
from constants import E
from logger import LogLevels, init_logger

DISDRO_TYPES = ["OTT HydroMet Parsivel2"]
WS_TYPES = ["Generic weather-station"]
RADAR_TYPES = ["BASTA", "METEK MIRA-35"]


@click.group()
@click.option("-v", "verbosity", count=True)
def cli(verbosity):
    log_level = LogLevels.get_by_verbosity_count(verbosity)
    init_logger(log_level)


@cli.command()
def status():
    print("Good, thanks !")


@cli.command()
# @click.option("--disdro-type", type=click.Choice(DISDRO_TYPES), required=True)
@click.option("--disdro-file", type=click.Path(exists=True), required=True)
# @click.option("--ws-type", type=click.Choice(WS_TYPES), required=True)
@click.option("--ws-file", type=click.Path(exists=True), required=True)
@click.option("--radar-file", type=click.Path(exists=True), required=True)
@click.option("--config-file", type=click.Path(exists=True), required=True)
@click.argument("output-file", type=click.Path())
def preprocess(disdro_file, ws_file, radar_file, config_file, output_file):
    """Command line interface for ccres_disdrometer_processing."""
    click.echo("CCRES disdrometer preprocessing : test CLI")

    config = toml.load(config_file)

    axrMethod = config["methods"]["AXIS_RATIO_METHOD"]
    strMethod = config["methods"]["FALL_SPEED_METHOD"]
    mieMethod = config["methods"]["COMPUTE_MIE_METHOD"]  # pymiecoated OR pytmatrix
    normMethod = config["methods"]["NORMALIZATION_METHOD"]  # measurement OR model
    beam_orientation = config["methods"]["BEAM_ORIENTATION"]

    # read weather-station data
    # ---------------------------------------------------------------------------------
    weather_xr = weather.read_weather_cloudnet(ws_file)

    # read doppler radar data
    # ---------------------------------------------------------------------------------
    radar_xr = radar.read_radar_cloudnet(radar_file)

    # read and preprocess disdrometer data
    # ---------------------------------------------------------------------------------

    disdro_xr = disdro.read_parsivel_cloudnet_choice(disdro_file)
    scatt = scattering.scattering_prop(
        disdro_xr.size_classes[0:-5],
        beam_orientation,
        radar_xr["lambda"].data,
        E,
        axrMethod=axrMethod,
        mieMethod=mieMethod,
    )
    disdro_xr = disdro.reflectivity_model(
        disdro_xr,
        scatt,
        len(disdro_xr.size_classes[0:-5]),
        radar_xr["lambda"].data,
        strMethod=strMethod,
        mieMethod=mieMethod,
        normMethod=normMethod,
    )
    print(
        weather_xr.time.values.shape,
        disdro_xr.time.values.shape,
        radar_xr.time.values.shape,
    )
    final_data = xr.merge([weather_xr, disdro_xr, radar_xr])

    final_data.to_netcdf(output_file)
    print(final_data.time.values.shape)
    return
