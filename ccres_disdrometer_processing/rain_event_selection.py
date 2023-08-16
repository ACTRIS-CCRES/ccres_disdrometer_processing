# import datetime as dt  # dates
import glob  # to load several files
import os

# import matplotlib.pyplot as plt  # to plot event statistics ?
import numpy as np
import pandas as pd  # save list of events as a csv file
import xarray as xr  # open CLU weather station data

STATION_NAME = "palaiseau"  # mettre en fichier de config !

DATA_PATH = (
    "/home/ygrit/Documents/disdro_processing/ccres_disdrometer_processing/"
    "daily_data/{}/weather-station".format(STATION_NAME)
)
# PATH_FILES_weather = DATA_PATH + "/*_{}_weather-station.nc".format(STATION_NAME)
PATH_FILES_weather = DATA_PATH + "/*.nc"
print(PATH_FILES_weather)
DB_PATH = (
    "/home/ygrit/Documents/disdro_processing/ccres_disdrometer_processing/"
    "bdd_rain_events/{}".format(STATION_NAME)
)

# Faire des path, des choix de configuration !
#  Pour rendre le code le plus générique possible.

CLIC = 0.2  # mm/mn : pluvio sampling. Varies between the different weather stations ?
RRMAX = 3  # mm/h
MIN_CUM = 1  # mm/episode
MAX_MEANWS = 7
MAX_WS = 10
MIN_T = 2  # °C

DELTA_EVENT = 60
DELTA_LENGTH = 180
CHUNK_THICKNESS = 15  # mn

FILES_weather = sorted(glob.glob(PATH_FILES_weather))
print(len(FILES_weather))


def selection():
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)

    weather_ds_full = xr.concat(
        (xr.open_dataset(file) for file in FILES_weather[:]), dim="time"
    )

    weather_ds_full["time"] = weather_ds_full.time.dt.round(freq="S")
    weather_ds_full["rain"] = weather_ds_full["rainfall_rate"] * 1000 * 60

    weather_ds = weather_ds_full.isel(
        {"time": np.where(weather_ds_full.rain.values >= CLIC)[0]}
    )
    weather_ds["rain_sum"] = np.cumsum(weather_ds["rain"])

    t = weather_ds.time

    start = []
    end = []

    start_candidate = t[0]

    for i in range(len(t) - 1):
        if t[i + 1] - t[i] >= np.timedelta64(DELTA_EVENT, "m"):
            if t[i] - start_candidate >= np.timedelta64(DELTA_LENGTH, "m"):
                start.append(start_candidate.values)
                end.append(t[i].values)
            start_candidate = t[i + 1]

    mask = np.ones(
        (len(start), 5), dtype=int
    )  # ordre : Min temperature, min rain accumulation, max wind avg, max wind, max RR
    test_values = np.empty((len(start), 6), dtype=object)

    # Overall constraints/flags on weather conditions

    for k in range(len(start)):
        min_temperature = (
            weather_ds_full["air_temperature"]
            .sel({"time": slice(start[k], end[k])})
            .min()
        )
        test_values[k, 0] = np.round(min_temperature.values - 273.15, decimals=2)
        if min_temperature < MIN_T:
            mask[k, 0] = 0

        accumulation = (
            weather_ds["rain_sum"].sel({"time": end[k]})
            - weather_ds["rain_sum"].loc[start[k]]
            + CLIC
        )  # add 1 trigger
        test_values[k, 1] = np.round(accumulation.values, decimals=2)
        if accumulation < MIN_CUM:
            mask[k, 1] = 0

        avg_ws = (
            weather_ds_full["wind_speed"].sel({"time": slice(start[k], end[k])}).mean()
        )
        test_values[k, 2] = np.round(avg_ws.values, decimals=2)
        if avg_ws > MAX_MEANWS:
            mask[k, 2] = 0

        max_ws = (
            weather_ds_full["wind_speed"].sel({"time": slice(start[k], end[k])}).max()
        )
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
                weather_ds["rain"]
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
        if RR_chunks_max > 3:
            mask[k, 4] = 0

        test_values[k, 5] = np.round(
            (
                weather_ds["rain_sum"].sel({"time": end[k]})
                - weather_ds["rain_sum"].sel({"time": start[k]})
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

    print(len(Events))
    Events.to_csv(
        DB_PATH
        + "/rain_events_{}_length{}_event{}.csv".format(
            STATION_NAME, DELTA_LENGTH, DELTA_EVENT
        ),
        header=True,
    )


if __name__ == "__main__":
    selection()
