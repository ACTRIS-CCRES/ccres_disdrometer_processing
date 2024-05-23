import glob

import numpy as np
import pandas as pd
import toml
import xarray as xr

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


def extract_timesteps_1day(file, rng, count):
    daily_ds = xr.open_dataset(file)[TIME_VARS + EVENTS_STARTEND].sel(
        {"range": rng}, method="nearest"
    )
    daily_filter = np.where(
        (daily_ds["flag_event"] > 0)
        & (daily_ds["QC_overall"] > 0)
        & (np.isfinite(daily_ds["Delta_Z"]))
    )[0]
    daily_df = daily_ds.isel({"time": daily_filter})[TIME_VARS].to_dataframe()
    daily_df["num_event"] = np.nan
    if len(daily_ds["events"]) > 0:
        for event in range(len(daily_ds["events"])):
            count += 1
            daily_df.loc[
                daily_ds["start_event"].values[event] : daily_ds["end_event"].values[
                    event
                ],
                ["num_event"],
            ] = int(count)
    return daily_df, count


def extract_1mn_events_data(folder, conf):
    files = sorted(glob.glob(folder))
    r = conf["instrument_parameters"]["DCR_DZ_RANGE"]  # range to keep for Delta_Z
    cpt = 0  # count for event number when looping over several daily processed files
    df_list = []
    for file in files:
        daily_df, cpt = extract_timesteps_1day(file=file, rng=r, count=cpt)
        df_list.append(daily_df)
    df = pd.concat(df_list)
    df = df.drop(columns=["range", "flag_event", "QC_overall"])
    return df


if __name__ == "__main__":
    folder = "/home/ygrit/disdro_processing/ccres_disdrometer_processing/tests/data/outputs/juelich_2021-12*_processed.nc"  # noqa
    conf = toml.load(
        "/home/ygrit/disdro_processing/ccres_disdrometer_processing/tests/data/conf/config_juelich_mira-parsivel.toml"
    )
    # stats_df = extract_stat_events(folder)
    # print(stats_df)
    timestep_df = extract_1mn_events_data(folder, conf)
    print(timestep_df)
