import datetime as dt
import glob
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

pd.options.mode.chained_assignment = None  # default='warn'

lgr = logging.getLogger(__name__)


CHUNK_THICKNESS = 15  # minutes

MN = 60  # mn : Time horizon before and after therain event
MAX_WS = 7  # m/s : Wind speed threshold for QC/QF
MIN_T = 2  # °C : Min temperature threshold for QC/QF
MIN_CUM = 3  # mm : minimum rain accumulation to keep an event in the statistics
MAX_RR = 3

CUM_REL_ERROR = 0.3  # 1, max relative error in rain accumulation measurement
FALLSPEED_REL_ERROR = (
    0.3  # 1, relative difference between theoretical and disdro fall speed
)

TIMESTAMP_THRESHOLDS = [MIN_T, MAX_WS, MAX_RR, FALLSPEED_REL_ERROR]
EVENT_THRESHOLDS = [MIN_CUM, CUM_REL_ERROR]

DELTA_DISDRO = dt.timedelta(minutes=MN)


def get_valid_paths(
    start: dt.datetime, end: dt.datetime, path: str
) -> Optional[list[str]]:
    lgr.info("Concatenate weather station data over the rain event duration")
    ndays = (
        (end + DELTA_DISDRO).replace(hour=0, minute=0, second=0)
        - (start - DELTA_DISDRO).replace(hour=0, minute=0, second=0)
    ).days
    dates = [(start - DELTA_DISDRO) + dt.timedelta(days=i) for i in range(ndays + 1)]

    paths = []
    for date in dates:
        files = glob.glob(path.format(date.strftime("%Y%m%d")))
        print("RADAR FILES WITH MATCHING NAMES : ", files)
        if len(files) == 0:
            lgr.critical("One file is missing")
            return None

        if not os.access(files[0], os.R_OK):
            lgr.critical(f"The file {files[0]} seems unreadable")
            return None

        lgr.info(f"station file {files[0]} found")
        paths.append(files[0])
    return paths


def data_pluvio_event(
    data_dir,
    start,
    end,
    threshold=TIMESTAMP_THRESHOLDS + EVENT_THRESHOLDS,
    main_wind_dir=270,
):  # start, end : datetimes.
    paths = get_valid_paths(start, end, f"{data_dir}/weather-station/{{}}*.nc")

    if paths is None:
        return None

    weather_ds = xr.concat((xr.open_dataset(path) for path in paths), dim="time")
    weather_ds["time"] = weather_ds.time.dt.round(freq="S")

    weather_event = weather_ds.sel(
        {"time": slice(start - DELTA_DISDRO, end + DELTA_DISDRO)}
    )
    weather_event["main_wind_dir"] = main_wind_dir
    weather_event["rain"] = weather_event["rainfall_rate"] * 1000 * 60
    weather_event["rain_sum"] = np.cumsum(weather_event["rain"])

    # Quality Flags
    weather_event["QF_T"] = weather_event["air_temperature"] > threshold[0] + 273.15
    weather_event["QF_ws"] = weather_event["wind_speed"] < threshold[1]
    weather_event["QF_wd"] = (
        np.abs(weather_event["wind_direction"]) - main_wind_dir < 45
    ) | (np.abs(weather_event["wind_direction"]) - (360 - main_wind_dir) < 45)
    weather_event["QF_acc"] = weather_event["rain_sum"] > threshold[4]
    weather_event["QF_RR"] = xr.DataArray(
        data=np.full(len(weather_event.time), True, dtype=bool), dims=["time"]
    )

    time_chunks = np.arange(
        np.datetime64(start), np.datetime64(end), np.timedelta64(CHUNK_THICKNESS, "m")
    )

    for start_time_chunk, stop_time_chunk in zip(time_chunks[:-1], time_chunks[1:]):
        RR_chunk = (
            weather_event["rain"]
            .sel(
                {
                    "time": slice(
                        start_time_chunk, stop_time_chunk - np.timedelta64(1, "m")
                    )
                }
            )
            .sum()
            * 60.0
            / CHUNK_THICKNESS
        )
        # weather_event["QF_RR"].sel({"time" : slice(start_time_chunk,time_chunks[j+1]
        # - np.timedelta64(1,'m'))}).values = np.tile((RR_chunk <= threshold[2]),
        # CHUNK_THICKNESS)
        time_slice = np.where(
            (weather_event.time >= start_time_chunk)
            & (weather_event.time <= stop_time_chunk - np.timedelta64(1, "m"))
        )
        weather_event["QF_RR"].values[time_slice] = np.tile(
            (RR_chunk <= threshold[2]), CHUNK_THICKNESS
        )

    return weather_event


def data_dcr_event(data_dir, start, end, r_type):
    paths = get_valid_paths(start, end, f"{data_dir}/radar/{{}}*{r_type}*.nc")
    # a été changé pour recevoir nom radar en .format dans get_valid_paths
    # dépannage temporaire ; à terme : DCR data extraite des fichiers préprocessés
    if paths is None:
        return None

    dcr_ds = xr.concat((xr.open_dataset(path) for path in paths), dim="time")
    dcr_ds["time"] = dcr_ds.time.dt.round(freq="S")

    dcr_event = dcr_ds.sel({"time": slice(start - DELTA_DISDRO, end + DELTA_DISDRO)})

    if dcr_event.time.size == 0:
        return None

    time_diffs = np.diff(dcr_event.time.values)
    negative_time_diffs = np.where(time_diffs / np.timedelta64(1, "s") < 0)[0]

    if len(negative_time_diffs) > 0:
        lgr.critical("DCR : time index problem")

        return None
    return dcr_event


def data_disdro_event(data_dir, start, end):
    # Path to disdro files is wrong for the moment !
    # The preprocessed data still needs to be generated and put somewhere easy
    # to compute tests with it
    paths = get_valid_paths(start, end, f"{data_dir}/disdrometer_preprocessed/{{}}*.nc")

    if paths is None:
        return None

    disdro_ds = xr.concat((xr.open_dataset(path) for path in paths), dim="time")
    disdro_ds["time"] = disdro_ds.time.dt.round(freq="S")

    disdro_event = disdro_ds.sel(
        {"time": slice(start - DELTA_DISDRO, end + DELTA_DISDRO)}
    )
    disdro_event["disdro_rain_sum"] = np.cumsum(disdro_event["RR"]) / 60

    return disdro_event


def get_data_event(
    data_dir,
    start,
    end,
    thresholds=TIMESTAMP_THRESHOLDS + EVENT_THRESHOLDS,
    main_wind_dir=270,
):
    weather = data_pluvio_event(data_dir, start, end, thresholds, main_wind_dir)
    dcr = data_dcr_event(data_dir, start, end)
    disdro = data_disdro_event(data_dir, start, end)

    return weather, dcr, disdro
