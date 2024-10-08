import glob
from dataclasses import dataclass

import numpy as np
import pandas as pd

# import toml
import xarray as xr


@dataclass
class DATA_DYN:
    dataframe: pd.DataFrame


TIME_VARS = ["Delta_Z", "flag_event", "QC_overall"]
EVENTS_STARTEND = ["start_event", "end_event"]


def extract_stat_events(folder):
    files = sorted(glob.glob(folder))
    file0 = xr.open_dataset(files[0])

    event_stats = []
    for var in list(file0.variables):
        if "events" in file0[var].dims:
            event_stats.append(var)

    ds = xr.concat([xr.open_dataset(file)[event_stats] for file in files], dim="events")
    ds.coords["events"] = np.arange(1, len(ds.events) + 1, 1)
    df = ds.to_dataframe()

    return df


def data_for_static_pdf(today, tomorrow, rng, min_timesteps):
    ds_to_concatenate = [
        xr.open_dataset(file)[
            TIME_VARS + EVENTS_STARTEND + ["QF_rg_dd_event"] + ["QF_rain_accumulation"]
        ].sel({"range": rng}, method="nearest")
        for file in [today, tomorrow]
    ]
    today_ds = ds_to_concatenate[0]
    ds = xr.concat(ds_to_concatenate, dim="time")
    data_event = []  # list of dataframes containing 1mn DeltaZ data for "eligible" events  # noqa E501
    cpt = 0
    weather_avail = ds.weather_data_avail.values[0]

    for s, e in zip(today_ds["start_event"].values, today_ds["end_event"].values):
        if today_ds["QF_rain_accumulation"].values[cpt] == 1 and (
            weather_avail == 0 or today_ds["QF_rg_dd_event"].values[cpt] == 1
        ):
            sub_ds = ds.sel(time=slice(s, e))
            sub_ds_nonan = sub_ds.sel(time=np.isfinite(sub_ds["Delta_Z"]))
            dz_r_good = sub_ds_nonan["Delta_Z"].isel(
                time=np.where(sub_ds_nonan["QC_overall"] == 1)[0]
            )
            if len(dz_r_good.values) >= min_timesteps:
                dz_r_good = dz_r_good.to_dataframe()
                dz_r_good["num_event"] = len(data_event) + 1
                data_event.append(dz_r_good)
        cpt += 1  # noqa: SIM113

    output = DATA_DYN(pd.concat(data_event + [pd.DataFrame()]))
    output.location = ds.attrs["location"]
    output.disdrometer_pid = ds.attrs["disdrometer_pid"]
    output.radar_pid = ds.attrs["radar_pid"]
    if "meteo_pid" in ds.attrs:
        output.meteo_pid = ds.attrs["meteo_pid"]
    return output
