import datetime as dt
import glob
import logging
import os

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import xarray as xr
from create_input_files_quicklooks import get_data_event
from matplotlib.gridspec import GridSpec
from rain_event_selection import selection
from scipy.optimize import curve_fit
from scipy.stats import cumfreq
from sklearn.linear_model import LinearRegression

CLIC = 0.2  # mm/mn : pluvio sampling. Varies between the different weather stations ?
RRMAX = 3  # mm/h
MIN_CUM = 1  # mm/episode
MAX_MEANWS = 7
MAX_WS = 10
MIN_T = 2  # °C

DELTA_EVENT = 60
DELTA_LENGTH = 180
CHUNK_THICKNESS = 15  # mn


def sel(
    list_preprocessed_files, delta_length=DELTA_LENGTH, delta_event=DELTA_EVENT
):  # lst : list of paths for DD preprocessed files
    preprocessed_ds_full = xr.concat(
        (xr.open_dataset(file) for file in list_preprocessed_files[:]), dim="time"
    )
    preprocessed_ds_full["time"] = preprocessed_ds_full.time.dt.round(freq="S")

    # preprocessed_ds = preprocessed_ds_full.isel(
    #     {"time": np.where(preprocessed_ds_full.rain.values >= CLIC)[0]}
    # )
    preprocessed_ds = preprocessed_ds_full.isel(
        {"time": np.where(preprocessed_ds_full.rain.values > 0)[0]}
    )

    preprocessed_ds["rain_cumsum"] = np.cumsum(preprocessed_ds_full["rain"])

    t = preprocessed_ds.time

    print(t, len(list_preprocessed_files))

    start = []
    end = []

    start_candidate = t[0]

    for i in range(len(t) - 1):
        if t[i + 1] - t[i] >= np.timedelta64(delta_event, "m"):
            if t[i] - start_candidate >= np.timedelta64(delta_length, "m"):
                start.append(start_candidate.values)
                end.append(t[i].values)
            start_candidate = t[i + 1]

    print(start, end)

    mask = np.ones(
        (len(start), 5), dtype=int
    )  # ordre : Min temperature, min rain accumulation, max wind avg, max wind, max RR
    test_values = np.empty((len(start), 6), dtype=object)

    # Overall constraints/flags on weather conditions

    for k in range(len(start)):
        min_temperature = (
            preprocessed_ds_full["temp"].sel({"time": slice(start[k], end[k])}).min()
        )
        test_values[k, 0] = np.round(min_temperature.values - 273.15, decimals=2)
        if min_temperature < MIN_T:
            mask[k, 0] = 0

        accumulation = (
            preprocessed_ds["rain_cumsum"].sel({"time": end[k]})
            - preprocessed_ds["rain_cumsum"].loc[start[k]]
            + CLIC
        )  # add 1 trigger
        test_values[k, 1] = np.round(accumulation.values, decimals=2)
        if accumulation < MIN_CUM:
            mask[k, 1] = 0

        avg_ws = (
            preprocessed_ds_full["ws"].sel({"time": slice(start[k], end[k])}).mean()
        )
        test_values[k, 2] = np.round(avg_ws.values, decimals=2)
        if avg_ws > MAX_MEANWS:
            mask[k, 2] = 0

        max_ws = preprocessed_ds_full["ws"].sel({"time": slice(start[k], end[k])}).max()
        test_values[k, 3] = np.round(max_ws.values, decimals=2)
        if max_ws > MAX_WS:
            mask[k, 3] = 0

        # Condition sur le taux de pluie max : faire un découpage de l'intvl de temps
        # en tronçons de 20mn et évaluer le taux de pluie sur ces tronçons.
        # Si ok pour tous les tronçons, alors événement OK.
        time_chunks = np.arange(
            np.datetime64(start[k]),
            np.datetime64(end[k]),
            np.timedelta64(CHUNK_THICKNESS, "m"),
        )
        RR_chunks = np.zeros(len(time_chunks) - 1)
        for j in range(len(time_chunks) - 1):
            RR_chunk = (
                preprocessed_ds["rain"]
                .sel(
                    {
                        "time": slice(
                            time_chunks[j], time_chunks[j + 1] - np.timedelta64(1, "m")
                        )
                    }
                )
                .sum()
                .values
                * 60.0
                / CHUNK_THICKNESS
            )  # mm/h
            RR_chunks[j] = RR_chunk
        RR_chunks_max = np.max(RR_chunks)
        test_values[k, 4] = np.round(RR_chunks_max.mean(), 3)
        if RR_chunks_max > RRMAX:
            mask[k, 4] = 0

        test_values[k, 5] = np.round(
            (
                preprocessed_ds["rain_cumsum"].sel({"time": end[k]})
                - preprocessed_ds["rain_cumsum"].sel({"time": start[k]})
            )
            / ((end[k] - start[k]) / np.timedelta64(1, "h")),
            decimals=2,
        ).values

    Events = pd.DataFrame(
        {
            "Start_time": start,
            "End_time": end,
            "Min Temperature (°C)": test_values[:, 0],
            "Rain accumulation (mm)": test_values[:, 1],
            "Avg WS (m/s)": test_values[:, 2],
            "Max WS (m/s)": test_values[:, 3],
            "max RR / {}mn subper (mm/h)".format(CHUNK_THICKNESS): test_values[:, 4],
            "avg RR (mm/h)": test_values[:, 5],
            "Min Temperature < {}°C".format(MIN_T): mask[:, 0],
            "Rain accumulation > {}mm".format(MIN_CUM): mask[:, 1],
            "Avg WS < {}m/s ?".format(MAX_MEANWS): mask[:, 2],
            "Max WS < {}m/s".format(MAX_WS): mask[:, 3],
            "Max Rain Rate <= {}mm".format(RRMAX): mask[:, 4],
        }
    )

    return Events


def dd_ql(preprocessed_ds, events):
    for start, end in zip(events["Start_time"], events["End_time"]):
        # ds = preprocessed_ds.sel({"time":slice(start, end)}) # +- DELTA_DISDRO


if __name__ == "__main__":
    path_ws_data = "/home/ygrit/Documents/disdro_processing/ccres_disdrometer_processing/daily_data/palaiseau/disdrometer_preprocessed/"
    lst = sorted(glob.glob(path_ws_data + "*.nc"))
    events = sel(lst)
    print(events)
