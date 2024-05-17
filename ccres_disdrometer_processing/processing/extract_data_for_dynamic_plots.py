import glob

import numpy as np
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


def extract_1mn_events_data(folder):
    pass


if __name__ == "__main__":
    folder = "/home/ygrit/disdro_processing/ccres_disdrometer_processing/tests/data/outputs/juelich_2021-12*_processed.nc"  # noqa
    df = extract_stat_events(folder)
    print(df)
