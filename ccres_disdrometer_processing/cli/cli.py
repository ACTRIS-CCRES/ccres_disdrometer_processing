"""Console script for ccres_disdrometer_processing."""

import sys
from pathlib import Path

import click
import toml
import xarray as xr
import numpy as np

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
@click.option("--disdro-file", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path, resolve_path=True, readable=True), required=True)
# @click.option("--ws-type", type=click.Choice(WS_TYPES), required=True)
@click.option("--ws-file", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path, resolve_path=True, readable=True), required=False, default=None) 
@click.option("--radar-file", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path, resolve_path=True, readable=True), required=True)
@click.option("--config-file", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path, resolve_path=True, readable=True), required=True)
@click.argument("output-file", type=click.Path(file_okay=True, dir_okay=False, path_type=Path, resolve_path=True))
# Add access controls so that click prépares the work ;
# give default value for WS file which is not mandatory
# resolve_path=True (chemin absolu) ; 
# path_type=True : va faire des path des objets "pathlib.Path"
# readable=True, writeable ? doesn't matter
# dir_ok=False, file_ok = True (I want a file and not a folder)
def preprocess(disdro_file, ws_file, radar_file, config_file, output_file):
    """Command line interface for ccres_disdrometer_processing."""
    click.echo("CCRES disdrometer preprocessing : test CLI")

    config = toml.load(config_file)

    axrMethod = config["methods"]["AXIS_RATIO_METHOD"]
    strMethod = config["methods"]["FALL_SPEED_METHOD"]
    mieMethod = config["methods"]["COMPUTE_MIE_METHOD"]  # pymiecoated OR pytmatrix
    normMethod = config["methods"]["NORMALIZATION_METHOD"]  # measurement OR model
    beam_orientation = config["methods"]["BEAM_ORIENTATION"]
    computed_frequencies = config["methods"]["COMPUTED_FREQUENCIES"] # given in Hz -> ok for the scattering script
    multilambda = bool(config["methods"]["multilambda"]) # True if multi lambda needed, False if only computation at lambda_radar needed

    # read doppler radar data
    # ---------------------------------------------------------------------------------
    radar_xr = radar.read_radar_cloudnet(radar_file)
    radar_frequency = radar_xr.frequency  # value is given in Hz in radar_xr file

    # read and preprocess disdrometer data
    # ---------------------------------------------------------------------------------

    disdro_xr = disdro.read_parsivel_cloudnet_choice(disdro_file, computed_frequencies)

    if multilambda == False :
        scatt = scattering.scattering_prop(
            disdro_xr.size_classes[0:-5],
            beam_orientation,
            radar_frequency,
            E,
            axrMethod=axrMethod,
            mieMethod=mieMethod,
        )
        disdro_xr = disdro.reflectivity_model(
            disdro_xr,
            scatt,
            len(disdro_xr.size_classes[0:-5]),
            radar_frequency,
            strMethod=strMethod,
            mieMethod=mieMethod,
            normMethod=normMethod,
        )
    else : 
        scatt_list = []
        for frequency in computed_frequencies :
            scatt = scattering.scattering_prop(
            disdro_xr.size_classes[0:-5],
            beam_orientation,
            frequency,
            E,
            axrMethod=axrMethod,
            mieMethod=mieMethod,
            )
            scatt_list.append(scatt)
        disdro_xr = disdro.reflectivity_model_multilambda(
            disdro_xr,
            scatt_list,
            len(disdro_xr.size_classes[0:-5]),
            np.array(computed_frequencies),
            strMethod=strMethod,
            mieMethod=mieMethod,
            normMethod=normMethod,
        )

    # read weather-station data
    # ---------------------------------------------------------------------------------
    if not (ws_file is None):
        weather_xr = weather.read_weather_cloudnet(ws_file)
        # print(list(weather_xr.keys()))
        final_data = xr.merge(
            [weather_xr, disdro_xr, radar_xr], combine_attrs="drop_conflicts"
        )
    else :
        final_data = xr.merge(
            [disdro_xr, radar_xr], combine_attrs="drop_conflicts"
        )

    weather_avail = int((not (weather is None)))
    final_data.attrs["weather_data_avail"] = weather_avail
    final_data.attrs["axis_ratioMethod"] = axrMethod
    final_data.attrs["fallspeedFormula"] = strMethod
    final_data.attrs["scatteringMethod"] = mieMethod
    final_data.attrs["DSDnormalizationMethod"] = normMethod
    final_data.attrs["beam_orientation"] = beam_orientation
    final_data.attrs["multilambda"] = int(multilambda)

    final_data.to_netcdf(output_file)
    print("OK")

    sys.exit(0) # Returns 0 if the code ran well
