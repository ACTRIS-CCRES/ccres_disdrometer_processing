"""Console script for ccres_disdrometer_processing."""

import sys
from pathlib import Path

import click
import toml
import xarray as xr
import numpy as np
import pandas as pd
import datetime
import subprocess

sys.path.append(str(Path(__file__).parent.parent))

import open_disdro_netcdf as disdro
import open_radar_netcdf as radar
import open_weather_netcdf as weather
import scattering as scattering
from __init__ import __version__
from constants import E
from logger import LogLevels, init_logger

import logging
lgr = logging.getLogger(__name__)

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

def preprocess(disdro_file, ws_file, radar_file, config_file, output_file):
    """Command line interface for ccres_disdrometer_processing."""
    click.echo("CCRES disdrometer preprocessing : test CLI")

    config = toml.load(config_file)

    axrMethod = config["methods"]["AXIS_RATIO_METHOD"]
    strMethod = config["methods"]["FALL_SPEED_METHOD"]
    mieMethod = config["methods"]["COMPUTE_MIE_METHOD"]  # pymiecoated OR pytmatrix
    computed_frequencies = config["methods"]["COMPUTED_FREQUENCIES"] # given in Hz -> ok for the scattering script

    # read doppler radar data
    # ---------------------------------------------------------------------------------
    radar_xr = radar.read_radar_cloudnet(radar_file)
    radar_frequency = radar_xr.radar_frequency  # value is given in Hz in radar_xr file

    # read and preprocess disdrometer data
    # ---------------------------------------------------------------------------------

    disdro_xr = disdro.read_parsivel_cloudnet_choice(disdro_file, computed_frequencies, config)
 
    scatt_list = []
    for fov in [1, 0]: # 1 : vertical fov, 0 : horizontal fov
        for frequency in computed_frequencies :
            scatt = scattering.scattering_prop(
            disdro_xr.size_classes[0:-5],
            fov,
            frequency,
            E,
            axrMethod=axrMethod,
            mieMethod=mieMethod,
            )
            scatt_list.append(scatt)
    disdro_xr = disdro.reflectivity_model_multilambda_measmodV_hvfov(
        disdro_xr,
        scatt_list,
        len(disdro_xr.size_classes[0:-5]),
        np.array(computed_frequencies),
        strMethod=strMethod,
        mieMethod=mieMethod,
    )

    # read weather-station data
    # ---------------------------------------------------------------------------------
    if not (ws_file is None):
        weather_xr = weather.read_weather_cloudnet(ws_file)
        final_data = xr.merge(
            [weather_xr, disdro_xr, radar_xr], combine_attrs="drop_conflicts"
        )
    else :
        final_data = xr.merge(
            [disdro_xr, radar_xr], combine_attrs="drop_conflicts"
        )

    final_data.time.attrs["standard_name"] = "time"
    weather_avail = int((not (weather is None)))
    final_data.attrs["weather_data_avail"] = weather_avail
    final_data.attrs["axis_ratioMethod"] = axrMethod
    final_data.attrs["fallspeedFormula"] = strMethod
    final_data.attrs["scatteringMethod"] = mieMethod

    # Add global attributes specified in the file format
    final_data.attrs["title"] = ""
    final_data.attrs["summary"] = ""
    final_data.attrs["keywords"] = "GCMD:EARTH SCIENCE, GCMD:ATMOSPHERE, GCMD:CLOUDS, GCMD:CLOUD DROPLET DISTRIBUTION, GCMD:CLOUD RADIATIVE TRANSFER, GCMD:CLOUD REFLECTANCE, GCMD:SCATTERING, GCMD:PRECIPITATION, GCMD:ATMOSPHERIC PRECIPITATION INDICES, GCMD:DROPLET SIZE, GCMD:HYDROMETEORS, GCMD:LIQUID PRECIPITATION, GCMD:RAIN, GCMD:LIQUID WATER EQUIVALENT, GCMD:PRECIPITATION AMOUNT, GCMD:PRECIPITATION RATE, GCMD:SURFACE PRECIPITATION"
    final_data.attrs["keywords_vocabulary"] = "GCMD:GCMD Keywords, CF:NetCDF COARDS Climate and Forecast Standard Names"
    final_data.attrs["Conventions"] = "CF-1.8, ACDD-1.3, GEOMS"
    final_data.attrs["id"] = config["nc_meta"]["id"]
    final_data.attrs["naming_authority"] = config["nc_meta"]["naming_authority"]
    date_created = datetime.datetime.now().isoformat()
    script_name = ""
    commit_id = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    final_data.attrs["history"] = "created on {} by {}, {}, {}".format(date_created,script_name,__version__,commit_id)
    final_data.attrs["source"] = "surface observation from {} DCR, {} disdrometer {}, processed by CloudNet".format(final_data.radar_source, final_data.disdrometer_source, "and AMS" * weather_avail)
    final_data.attrs["processing_level"] = "2a"
    final_data.attrs["comment"] = config["nc_meta"]["comment"]
    final_data.attrs["acknowledgement"] = ""
    final_data.attrs["license"] = "CC BY 4.0"
    final_data.attrs["standard_name_vocabulary"] = "CF Standard Name Table v84"
    final_data.attrs["date_created"] = date_created
    final_data.attrs["creator_name"] = config["nc_meta"]["creator_name"]
    final_data.attrs["creator_email"] = config["nc_meta"]["creator_email"]
    final_data.attrs["creator_url"] = config["nc_meta"]["creator_url"]
    final_data.attrs["creator_type"] = config["nc_meta"]["creator_type"]
    final_data.attrs["creator_institution"] = config["nc_meta"]["creator_institution"]
    final_data.attrs["institution"] = config["nc_meta"]["institution"]
    final_data.attrs["project"] = config["nc_meta"]["project"]
    final_data.attrs["publisher_name"] = config["nc_meta"]["publisher_name"]
    final_data.attrs["publisher_email"] = config["nc_meta"]["publisher_email"]
    final_data.attrs["publisher_url"] = config["nc_meta"]["publisher_url"]
    final_data.attrs["publisher_type"] = config["nc_meta"]["publisher_type"]
    final_data.attrs["publisher_institution"] = config["nc_meta"]["publisher_institution"]
    final_data.attrs["contributor_name"] = config["nc_meta"]["contributor_name"]
    final_data.attrs["contributor_role"] = config["nc_meta"]["contributor_role"]
    final_data.attrs["geospatial_bounds"] = "POLYGON"
    final_data.attrs["geospatial_bounds_crs"] = "EPSG:4326" # WGS84
    final_data.attrs["geospatial_bounds_vertical_crs"] = "EPSG:5829"
    final_data.attrs["geospatial_lat_min"] = np.min(np.array([final_data.disdro_latitude.values, final_data.radar_latitude.values, final_data.ams_latitude.values]))
    final_data.attrs["geospatial_lat_max"] = np.max(np.array([final_data.disdro_latitude.values, final_data.radar_latitude.values, final_data.ams_latitude.values]))
    final_data.attrs["geospatial_lat_units"] = "degree_north"
    def precision(nb):
        return str(nb)[::-1].find('.')
    final_data.attrs["geospatial_lat_resolution"] = 10**-(min(precision(final_data.disdro_latitude.values), precision(final_data.radar_latitude.values), precision(final_data.ams_latitude.values)))
    final_data.attrs["geospatial_lon_min"] = np.min(np.array([final_data.disdro_longitude.values, final_data.radar_longitude.values, final_data.ams_longitude.values]))
    final_data.attrs["geospatial_lon_max"] = np.max(np.array([final_data.disdro_longitude.values, final_data.radar_longitude.values, final_data.ams_longitude.values]))
    final_data.attrs["geospatial_lon_units"] = "degree_east"
    final_data.attrs["geospatial_lon_resolution"] = 10**-(min(precision(final_data.disdro_longitude.values), precision(final_data.radar_longitude.values), precision(final_data.ams_longitude.values)))
    final_data.attrs["geospatial_vertical_min"] = np.min(np.array([final_data.disdro_altitude.values, final_data.radar_altitude.values, final_data.ams_altitude.values]))
    final_data.attrs["geospatial_vertical_max"] = np.max(np.array([final_data.disdro_altitude.values, final_data.radar_altitude.values, final_data.ams_altitude.values]))
    final_data.attrs["geospatial_vertical_units"] = "m"
    final_data.attrs["geospatial_vertical_resolution"] = 10**-(min(precision(final_data.disdro_altitude.values), precision(final_data.radar_altitude.values), precision(final_data.ams_altitude.values)))
    final_data.attrs["geospatial_vertical_positive"] = "up"
    final_data.attrs["time_coverage_start"] = pd.Timestamp(final_data.time.values[0]).isoformat()
    final_data.attrs["time_coverage_end"] = pd.Timestamp(final_data.time.values[-1]).isoformat()
    final_data.attrs["time_coverage_duration"] = pd.Timedelta(final_data.time.values[-1] - final_data.time.values[0]).isoformat()
    final_data.attrs["time_coverage_resolution"] = pd.Timedelta(final_data.time.values[1] - final_data.time.values[0]).isoformat() # PT60S here
    final_data.attrs["program"] = "ACTRIS, CloudNet, CCRES"
    final_data.attrs["date_modified"] = date_created
    final_data.attrs["date_issued"] = date_created # made available immediately to the users after ceration
    final_data.attrs["date_metadata_modified"] = "" # will be set when everything will be of ; modify it if some fields evolve
    final_data.attrs["product_version"] = __version__
    final_data.attrs["platform"] = "GCMD:In Situ Land-based Platforms, GCMD:OBSERVATORIES"
    final_data.attrs["platform_vocabulary"] = "GCMD:GCMD Keywords"
    final_data.attrs["instrument"] = "GCMD:Earth Remote Sensing Instruments, GCMD:Active Remote Sensing, GCMD:Profilers/Sounders, GCMD:Radar Sounders, GCMD:DOPPLER RADAR, GCMD:FMCWR, GCMD:VERTICAL POINTING RADAR, GCMD:In Situ/Laboratory Instruments, GCMD:Gauges, GCMD:RAIN GAUGES, GCMD:Recorders/Loggers, GCMD:DISDROMETERS, GCMD:Temperature/Humidity Sensors, GCMD:TEMPERATURE SENSORS, GCMD:HUMIDITY SENSORS, GCMD:Current/Wind Meters, GCMD:WIND MONITOR, GCMD:Pressure/Height Meters, GCMD:BAROMETERS"
    final_data.attrs["instrument_vocabulary"] = "GCMD:GCMD Keywords"
    final_data.attrs["cdm_data_type"] = config["nc_meta"]["cdm_data_type"] # empty
    final_data.attrs["metadata_link"] = config["nc_meta"]["cdm_data_type"] # empty
    final_data.attrs["references"] = "" # empty for the moment ; add the reference quotation to the code if an article is published


























    final_data.to_netcdf(output_file)
    lgr.info("Preprocessing : SUCCESS")

    sys.exit(0) # Returns 0 if the code ran well
