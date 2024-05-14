import glob

import matplotlib.pyplot as plt
import xarray as xr


def monitoring_timeseries(folder):
    files = sorted(glob.glob(folder))
    print(len(files))
    ds = xr.concat([xr.open_dataset(file) for file in files], dim="events")
    print(ds.dims)

    fig, ax = plt.subplots(figsize=((20, 6)))

    return


if __name__ == "__main__":
    folder = "/bdd/ACTRIS/CCRES/pub/disdro/juelich/2024/*/*/*proc*.nc"
    monitoring_timeseries(folder)
