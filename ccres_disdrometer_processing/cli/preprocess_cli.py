import datetime
import logging
import sys

import click
import numpy as np
import pandas as pd
import toml
import xarray as xr

import ccres_disdrometer_processing.open_disdro_netcdf as disdro
import ccres_disdrometer_processing.open_radar_netcdf as radar
import ccres_disdrometer_processing.open_weather_netcdf as weather
import ccres_disdrometer_processing.scattering as scattering
from ccres_disdrometer_processing.__init__ import __version__, script_name
from ccres_disdrometer_processing.logger import LogLevels, init_logger

ISO_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
TIME_UNITS = "seconds since 2000-01-01T00:00:00.0Z"
TIME_CALENDAR = "standard"

lgr = logging.getLogger(__name__)


def preprocess(disdro_file, ws_file, radar_file, config_file, output_file, verbosity):
    """Command line interface for ccres_disdrometer_processing."""
    log_level = LogLevels.get_by_verbosity_count(verbosity)
    init_logger(log_level)
    click.echo("CCRES disdrometer preprocessing : test CLI")

    lgr.info("Load configuration file")
    config = toml.load(config_file)

    axrMethod = config["methods"]["AXIS_RATIO_METHOD"]
    strMethod = config["methods"]["FALL_SPEED_METHOD"]
    refraction_indices = config["methods"]["REFRACTION_INDEX"]
    computed_frequencies = config["methods"][
        "RADAR_FREQUENCIES"
    ]  # given in Hz -> ok for the scattering script
    E = {}
    for frequency, index in zip(computed_frequencies, refraction_indices):
        refr_index = complex(index[0], index[1])
        E[frequency] = refr_index

    max_radar_altitude = config["methods"]["MAX_ALTITUDE_RADAR_DATA"]

    # read doppler radar data
    # ---------------------------------------------------------------------------------
    lgr.info("Open and read radar data")
    radar_xr = radar.read_radar_cloudnet(radar_file, max_radar_alt=max_radar_altitude)
    lgr.info("Radar data : OK")

    # read and preprocess disdrometer data
    # ---------------------------------------------------------------------------------
    lgr.info("Open and read disdrometer raw data")
    disdro_xr = disdro.read_parsivel_cloudnet_choice(
        disdro_file, computed_frequencies, config
    )
    lgr.info("Disdrometer raw data : OK")

    lgr.info("Begin computation of forward modeled-reflectivity and DSD parameters")
    scatt_list = []
    for fov in [1, 0]:  # 1 : vertical fov, 0 : horizontal fov
        for frequency in computed_frequencies:
            scatt = scattering.scattering_prop(
                disdro_xr.size_classes[0:-5],
                fov,
                frequency,
                e=E[frequency],
                axrMethod=axrMethod,
            )
            scatt_list.append(scatt)
    disdro_xr = disdro.reflectivity_model_multilambda_measmodV_hvfov(
        disdro_xr,
        scatt_list,
        len(disdro_xr.size_classes[0:-5]),
        strMethod=strMethod,
    )
    lgr.info("Reflectivity and DSD parameters : OK")

    # read weather-station data
    # ---------------------------------------------------------------------------------
    weather_avail = ws_file is not None
    if weather_avail:
        lgr.info("Weather data provided")
        lgr.info("Open and read weather data")
        weather_xr = weather.read_weather_cloudnet(ws_file)
        lgr.info("Weather data : OK")
        lgr.info("Merge weather, disdrometer and radar data")
        final_data = xr.merge(
            [weather_xr, disdro_xr, radar_xr], combine_attrs="drop_conflicts"
        )
        lgr.info("Merge : OK")
    else:
        lgr.info("No weather data provided")
        lgr.info("Merge disdrometer and radar data")
        final_data = xr.merge([disdro_xr, radar_xr], combine_attrs="drop_conflicts")
        lgr.info("Merge : OK")

    final_data["weather_data_avail"] = np.array(weather_avail).astype("i2")
    final_data["weather_data_avail"].attrs["long_name"] = (
        "Availability of weather data at the station"
    )
    final_data["weather_data_avail"].attrs["flag_values"] = np.array([0, 1]).astype(
        "i2"
    )
    final_data["weather_data_avail"].attrs["flag_meanings"] = (
        "no_weather_file_available weather_file_provided"
    )

    lgr.info("Add netCDF missing global attributes")

    final_data.attrs["station_name"] = config["location"]["STATION"]
    final_data.time.attrs["standard_name"] = "time"
    final_data.attrs["axis_ratioMethod"] = axrMethod
    final_data.attrs["fallspeedFormula"] = strMethod

    # Add global attributes specified in the file format
    final_data.attrs["title"] = (
        f"CCRES pre-processing file for Doppler cloud radar stability monitoring with disdrometer at {final_data.attrs['location']} site"  # noqa E501
    )
    final_data.attrs["summary"] = (
        f"Disdrometer ({final_data.attrs['disdrometer_source']}) data are processed to derive the equivalent reflectivity factor at {len(computed_frequencies)} frequencies ({', '.join(str(round(freq * 1e-9, 0)) for freq in computed_frequencies[:])} GHz). Doppler cloud radar ({final_data.attrs['radar_source']}) data (reflectivity and Doppler velocity) are extracted up to some hundreds of meters, and weather station data (temperature, humidity, wind and precipitation rate) are added to the dataset if provided. The resulting pre-processing netCDF file has a 1-minute sampling for all the collocated sensors."  # noqa E501
    )
    final_data.attrs["keywords"] = (
        "GCMD:EARTH SCIENCE, GCMD:ATMOSPHERE, GCMD:CLOUDS, GCMD:CLOUD DROPLET DISTRIBUTION, GCMD:CLOUD RADIATIVE TRANSFER, GCMD:CLOUD REFLECTANCE, GCMD:SCATTERING, GCMD:PRECIPITATION, GCMD:ATMOSPHERIC PRECIPITATION INDICES, GCMD:DROPLET SIZE, GCMD:HYDROMETEORS, GCMD:LIQUID PRECIPITATION, GCMD:RAIN, GCMD:LIQUID WATER EQUIVALENT, GCMD:PRECIPITATION AMOUNT, GCMD:PRECIPITATION RATE, GCMD:SURFACE PRECIPITATION"  # noqa
    )
    final_data.attrs["keywords_vocabulary"] = (
        "GCMD:GCMD Keywords, CF:NetCDF COARDS Climate and Forecast Standard Names"
    )
    final_data.attrs["Conventions"] = "CF-1.8, ACDD-1.3, GEOMS"
    final_data.attrs["id"] = config["nc_meta"]["id"]
    final_data.attrs["naming_authority"] = config["nc_meta"]["naming_authority"]
    date_created = datetime.datetime.utcnow().strftime(ISO_DATE_FORMAT)
    final_data.attrs["history"] = (
        f"created on {date_created} by {script_name}, v{__version__}"
    )
    weather_str = ""
    if weather_avail:
        weather_str = " and AMS"
    final_data.attrs["source"] = (
        f"surface observation from {final_data.radar_source} DCR, {final_data.disdrometer_source} disdrometer{weather_str}, processed by CloudNet"  # noqa
    )
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
    final_data.attrs["publisher_institution"] = config["nc_meta"][
        "publisher_institution"
    ]
    final_data.attrs["contributor_name"] = config["nc_meta"]["contributor_name"]
    final_data.attrs["contributor_role"] = config["nc_meta"]["contributor_role"]

    def precision(nb):
        return str(nb)[::-1].find(".")

    if weather_avail:
        geospatial_lat_min = np.nanmin(
            [
                np.nanmin(final_data.disdro_latitude.values),
                np.nanmin(final_data.radar_latitude.values),
                np.nanmin(final_data.ams_latitude.values),
            ]
        )
        geospatial_lat_max = np.nanmax(
            [
                np.nanmax(final_data.disdro_latitude.values),
                np.nanmax(final_data.radar_latitude.values),
                np.nanmax(final_data.ams_latitude.values),
            ]
        )
        geospatial_lon_min = np.nanmin(
            [
                np.nanmin(final_data.disdro_longitude.values),
                np.nanmin(final_data.radar_longitude.values),
                np.nanmin(final_data.ams_longitude.values),
            ]
        )
        geospatial_lon_max = np.nanmax(
            [
                np.nanmax(final_data.disdro_longitude.values),
                np.nanmax(final_data.radar_longitude.values),
                np.nanmax(final_data.ams_longitude.values),
            ]
        )
        geospatial_lat_res = 10 ** -(
            min(
                precision(np.nanmin(final_data.disdro_latitude.values)),
                precision(np.nanmin(final_data.radar_latitude.values)),
                precision(np.nanmin(final_data.ams_latitude.values)),
            )
        )
        geospatial_lon_res = 10 ** -(
            min(
                precision(np.nanmin(final_data.disdro_longitude.values)),
                precision(np.nanmin(final_data.radar_longitude.values)),
                precision(np.nanmin(final_data.ams_longitude.values)),
            )
        )
        geospatial_vert_min = np.nanmin(
            [
                np.nanmin(final_data.disdro_altitude.values),
                np.nanmin(final_data.radar_altitude.values),
                np.nanmin(final_data.ams_altitude.values),
            ]
        )
        geospatial_vert_max = np.nanmax(
            [
                np.nanmax(final_data.disdro_altitude.values),
                np.nanmax(final_data.radar_altitude.values),
                np.nanmax(final_data.ams_altitude.values),
            ]
        )
        geospatial_vert_res = 10 ** -(
            min(
                precision(np.nanmin(final_data.disdro_altitude.values)),
                precision(np.nanmin(final_data.radar_altitude.values)),
                precision(np.nanmin(final_data.ams_altitude.values)),
            )
        )
    else:
        geospatial_lat_min = np.nanmin(
            [
                np.nanmin(final_data.disdro_latitude.values),
                np.nanmin(final_data.radar_latitude.values),
            ]
        )
        geospatial_lat_max = np.nanmax(
            [
                np.nanmax(final_data.disdro_latitude.values),
                np.nanmax(final_data.radar_latitude.values),
            ]
        )
        geospatial_lon_min = np.nanmin(
            [
                np.nanmin(final_data.disdro_longitude.values),
                np.nanmin(final_data.radar_longitude.values),
            ]
        )

        geospatial_lon_max = np.nanmax(
            [
                np.nanmax(final_data.disdro_longitude.values),
                np.nanmax(final_data.radar_longitude.values),
            ]
        )
        geospatial_lat_res = 10 ** -(
            min(
                precision(np.nanmin(final_data.disdro_latitude.values)),
                precision(np.nanmin(final_data.radar_latitude.values)),
            )
        )
        geospatial_lon_res = 10 ** -(
            min(
                precision(np.nanmin(final_data.disdro_longitude.values)),
                precision(np.nanmin(final_data.radar_longitude.values)),
            )
        )
        geospatial_vert_min = np.nanmin(
            [
                np.nanmin(final_data.disdro_altitude.values),
                np.nanmin(final_data.radar_altitude.values),
            ]
        )
        geospatial_vert_max = np.nanmax(
            [
                np.nanmax(final_data.disdro_altitude.values),
                np.nanmax(final_data.radar_altitude.values),
            ]
        )
        geospatial_vert_res = 10 ** -(
            min(
                precision(np.nanmin(final_data.disdro_altitude.values)),
                precision(np.nanmin(final_data.radar_altitude.values)),
            )
        )
    final_data.attrs["geospatial_bounds"] = (
        f"POLYGON (({geospatial_lat_min}, {geospatial_lon_min}), ({geospatial_lat_min}, {geospatial_lon_max}), ({geospatial_lat_max}, {geospatial_lon_max}), ({geospatial_lat_max}, {geospatial_lon_min}))"  # noqa
    )
    final_data.attrs["geospatial_bounds_crs"] = "EPSG:4326"  # WGS84
    final_data.attrs["geospatial_bounds_vertical_crs"] = "EPSG:5829"
    final_data.attrs["geospatial_lat_min"] = geospatial_lat_min
    final_data.attrs["geospatial_lat_max"] = geospatial_lat_max
    final_data.attrs["geospatial_lat_units"] = "degree_north"
    final_data.attrs["geospatial_lat_resolution"] = geospatial_lat_res
    final_data.attrs["geospatial_lon_min"] = geospatial_lon_min
    final_data.attrs["geospatial_lon_max"] = geospatial_lon_max
    final_data.attrs["geospatial_lon_units"] = "degree_east"
    final_data.attrs["geospatial_lon_resolution"] = geospatial_lon_res
    final_data.attrs["geospatial_vertical_min"] = geospatial_vert_min
    final_data.attrs["geospatial_vertical_max"] = geospatial_vert_max
    final_data.attrs["geospatial_vertical_units"] = "m"
    final_data.attrs["geospatial_vertical_resolution"] = geospatial_vert_res
    final_data.attrs["geospatial_vertical_positive"] = "up"

    final_data.attrs["time_coverage_start"] = pd.Timestamp(
        final_data.time.values[0]
    ).strftime(ISO_DATE_FORMAT)
    final_data.attrs["time_coverage_end"] = pd.Timestamp(
        final_data.time.values[-1]
    ).strftime(ISO_DATE_FORMAT)
    final_data.attrs["time_coverage_duration"] = pd.Timedelta(
        final_data.time.values[-1] - final_data.time.values[0]
    ).isoformat()
    final_data.attrs["time_coverage_resolution"] = pd.Timedelta(
        final_data.time.values[1] - final_data.time.values[0]
    ).isoformat()  # PT60S here
    final_data.attrs["program"] = "ACTRIS, CloudNet, CCRES"
    final_data.attrs["date_modified"] = date_created
    final_data.attrs["date_issued"] = (
        date_created  # made available immediately to the users after ceration
    )
    final_data.attrs["date_metadata_modified"] = (
        ""  # will be set when everything will be of ; modify it if some fields evolve
    )
    final_data.attrs["product_version"] = __version__
    final_data.attrs["platform"] = (
        "GCMD:In Situ Land-based Platforms, GCMD:OBSERVATORIES"
    )
    final_data.attrs["platform_vocabulary"] = "GCMD:GCMD Keywords"
    final_data.attrs["instrument"] = (
        "GCMD:Earth Remote Sensing Instruments, GCMD:Active Remote Sensing, GCMD:Profilers/Sounders, GCMD:Radar Sounders, GCMD:DOPPLER RADAR, GCMD:FMCWR, GCMD:VERTICAL POINTING RADAR, GCMD:In Situ/Laboratory Instruments, GCMD:Gauges, GCMD:RAIN GAUGES, GCMD:Recorders/Loggers, GCMD:DISDROMETERS, GCMD:Temperature/Humidity Sensors, GCMD:TEMPERATURE SENSORS, GCMD:HUMIDITY SENSORS, GCMD:Current/Wind Meters, GCMD:WIND MONITOR, GCMD:Pressure/Height Meters, GCMD:BAROMETERS"  # noqa
    )
    final_data.attrs["instrument_vocabulary"] = "GCMD:GCMD Keywords"
    final_data.attrs["cdm_data_type"] = config["nc_meta"]["cdm_data_type"]  # empty
    final_data.attrs["metadata_link"] = config["nc_meta"]["metadata_link"]  # empty
    # TODO: add the reference quotation to the code if an article is published
    final_data.attrs["references"] = ""

    lgr.info("netCDF global attributes : OK")
    lgr.info("Convert Dataset to netCDF file")

    final_data.to_netcdf(
        output_file, encoding={"time": {"units": TIME_UNITS, "calendar": "standard"}}
    )

    click.echo("Preprocessing : SUCCESS")

    sys.exit(0)  # Returns 0 if the code ran well
