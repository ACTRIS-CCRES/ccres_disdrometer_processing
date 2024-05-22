import glob

import numpy as np
import pandas as pd
import toml
import xarray as xr


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


def extract_1mn_events_data(folder, conf):
    files = sorted(glob.glob(folder))
    time_vars = ["Delta_Z", "flag_event", "QC_overall"]
    events_startend = ["start_event", "end_event"]
    r = conf["instrument_parameters"]["DCR_DZ_RANGE"]  # range to keep for Delta_Z
    daily_ds0 = xr.open_dataset(files[0])[time_vars + events_startend].sel(
        {"range": r}, method="nearest"
    )
    filter0 = np.where(
        (daily_ds0["flag_event"] > 0)
        & (daily_ds0["QC_overall"] > 0)
        & (np.isfinite(daily_ds0["Delta_Z"]))
    )[0]
    df = daily_ds0.isel({"time": filter0})[time_vars].to_dataframe()
    df["num_event"] = np.nan
    cpt = 0
    for event in range(len(daily_ds0["events"])):
        cpt += 1
        df.loc[
            daily_ds0["start_event"].values[event] : daily_ds0["end_event"].values[
                event
            ],
            ["num_event"],
        ] = int(cpt)

    for file in files:
        daily_ds = xr.open_dataset(file)[time_vars + events_startend].sel(
            {"range": r}, method="nearest"
        )
        daily_filter = np.where(
            (daily_ds["flag_event"] > 0)
            & (daily_ds["QC_overall"] > 0)
            & (np.isfinite(daily_ds["Delta_Z"]))
        )[0]
        daily_df = daily_ds.isel({"time": daily_filter})[time_vars].to_dataframe()
        daily_df["num_event"] = np.nan
        if len(daily_ds["events"]) > 0:
            for event in range(len(daily_ds["events"])):
                cpt += 1
                daily_df.loc[
                    daily_ds["start_event"].values[event] : daily_ds[
                        "end_event"
                    ].values[event],
                    ["num_event"],
                ] = int(cpt)
                df = pd.concat([df, daily_df])
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
