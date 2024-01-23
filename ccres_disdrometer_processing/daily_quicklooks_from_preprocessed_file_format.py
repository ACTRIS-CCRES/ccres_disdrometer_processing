"""
Input : output of the preprocessing = daily file
Output : 2 Panels of daily quicklooks : 2D DCR data, disdrometer information, weather variables if provided
"""


import logging
import os
from pathlib import Path

import numpy as np
import xarray as xr
import datetime as dt
import pandas as pd
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec

import scipy.optimize as so
import scipy.stats as stats

lgr = logging.getLogger(__name__)


def daily_quicklooks(preprocessed_file):
    ds = xr.open_dataset(preprocessed_file)

    # Date format for plots
    locator = mpl.dates.AutoDateLocator()
    formatter = mpl.dates.ConciseDateFormatter(locator)

    # Panel 1 : 2D radar data, disdrometer data, weather data (if ams file is provided)
    fig, axes = plt.subplots(3,2, figsize=(20,20))
    (ax1, ax2, ax3, ax4, ax5, ax6) = axes[0,0], axes[0,1], axes[1,0], axes[1,1], axes[2,0], axes[2,1]
    plt.subplots_adjust(0.06, 0.06, 0.93, 0.9, hspace=0.25, wspace=0.3)

    fig.text(
    s=pd.Timestamp(ds.time.values[0]).to_pydatetime().strftime("%Y-%m-%d") + " : Daily data overview",
    fontsize=18,
    horizontalalignment="center",
    verticalalignment="center",
    y=0.97,
    x=0.5,
)

    for axe in axes[0:5].flatten():
        axe.xaxis.set_major_formatter(formatter)
        axe.set_xlabel("Time (UTC)")

    # Plot 2D DCR Reflectivity
    cmap = plt.get_cmap("rainbow").copy()
    vmin_z, vmax_z = 0, 20
    cmap.set_under("w")
    cmap.set_over("r")

    pc = ax1.pcolormesh(
    ds.time.values,
    ds.range.values,
    ds.Zdcr.values.T,
    vmin=vmin_z,
    vmax=vmax_z,
    cmap=cmap,
    shading="nearest",
)
    ax1.set_ylabel(f"{ds.alt.attrs['long_name']} ({ds.alt.attrs['units']})")
    ax1.set_title("DCR {:.0f} GHz reflectivity".format(ds.radar_frequency*1e-9))
    pos = ax1.get_position()
    cb_ax = fig.add_axes([pos.x1 + 0.005, pos.y0, 0.015, pos.y1 - pos.y0])
    cb = fig.colorbar(pc, orientation="vertical", cax=cb_ax)
    cb.set_label(f"{ds.Zdcr.attrs['long_name']} ({ds.Zdcr.attrs['units']})")

    # Plot 2D DCR Doppler velocity
    cmap = plt.get_cmap("rainbow").copy()
    vmin_dv, vmax_dv = -5, 5
    cmap.set_under("w")
    cmap.set_over("w")

    pc = ax2.pcolormesh(
    ds.time.values,
    ds.range.values,
    ds.DVdcr.values.T,
    vmin=vmin_dv,
    vmax=vmax_dv,
    cmap=cmap,
    shading="nearest",
)
    ax2.set_ylabel(f"{ds.alt.attrs['long_name']} ({ds.alt.attrs['units']})")
    ax2.set_title("DCR {:.0f} Doppler Velocity".format(ds.radar_frequency*1e-9))
    pos = ax2.get_position()
    cb_ax = fig.add_axes([pos.x1 + 0.005, pos.y0, 0.015, pos.y1 - pos.y0])
    cb = fig.colorbar(pc, orientation="vertical", cax=cb_ax)
    cb.set_label(f"{ds.DVdcr.attrs['long_name']} ({ds.DVdcr.attrs['units']})")

    # 1.4. Air temperature / relative humidity time series
    ax3b = ax3.twinx()
    temperature = ax3.plot(ds.time, ds["ta"].values, label = ds["ta"].attrs["long_name"], color = "red")
    ax3.set_ylabel(f"{ds['ta'].attrs['long_name']} ({ds['ta'].attrs['units']})")

    humidity = ax3b.plot(ds.time, ds["hur"].values, label = ds["hur"].attrs["long_name"], color = "green")
    ax3b.set_ylabel(f"{ds['hur'].attrs['long_name']} ({ds['hur'].attrs['units']})")

    ax3.grid()
    ax3.set_title("Timeseries for Air temperature and Relative humidity")
    lines = temperature + humidity
    line_labels = [l.get_label() for l in lines]
    ax3.legend(lines, line_labels, loc='best')

    # Air temperature / relative humidity time series
    ax4b = ax4.twinx()
    wind_speed = ax4.plot(ds.time, ds["ws"].values, label = ds["ws"].attrs["long_name"], color = "red")
    ax4.set_ylabel(f"{ds['ws'].attrs['long_name']} ({ds['ws'].attrs['units']})")

    wind_direction = ax4b.plot(ds.time, ds["wd"].values, label = ds["wd"].attrs["long_name"], color = "green")
    ax4b.set_ylabel(f"{ds['wd'].attrs['long_name']} ({ds['wd'].attrs['units']})")

    ax4.grid()
    ax4.set_title("Timeseries for Wind speed and direction")
    lines = temperature + humidity
    line_labels = [l.get_label() for l in lines]
    ax4.legend(lines, line_labels, loc='best')

    # Disdrometer-related data : rain accumulation and 2D plot for the Speed/Diameter relationship
    ax5.plot(
        ds.time, ds["disdro_cp"], color="red", label="Disdrometer"
    )
    ax5.plot(
        ds.time[:],
        ds["ams_cp"][:],
        color="green",
        label="Weather station",
    ) 
    ax5.legend()
    ax5.set_ylabel("Cumulated precipitation [mm]")
    ax5.set_title("Disdrometer and weather station-based cumulated precipitation over the day")
    ax5.grid()

    ax6

    # ax5.plot(weather.time, weather["wind_speed"])
    # ax5.set_title("Wind Speed")
    # ax5.set_ylabel("wind speed [m/s]")
    # ax5.grid()

    # plt.show()

    return fig



if __name__ == "__main__" :
    MAIN_DIR = Path(__file__).parent.parent
    TEST_DIR = MAIN_DIR / "tests"
    TEST_OUT_DIR = TEST_DIR / "data/outputs"
    preprocessed_file = f"{TEST_OUT_DIR}/20210202_palaiseau_preprocessed_v1101.nc"
    fig = daily_quicklooks(preprocessed_file)
    fig.savefig(f"{TEST_OUT_DIR}/20210202_palaiseau_test_quicklooks.png", dpi=500, transparent=False)