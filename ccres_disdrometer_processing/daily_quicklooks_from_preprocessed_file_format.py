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

import toml

lgr = logging.getLogger(__name__)


def daily_quicklooks(preprocessed_file, config_file):
    ds = xr.open_dataset(preprocessed_file)
    config = toml.load(config_file)
    gate = config["methods"]["DZ_RANGE"]
    # Date format for plots
    locator = mpl.dates.AutoDateLocator()
    formatter = mpl.dates.ConciseDateFormatter(locator)

    fig1 = panel1(ds, locator, formatter)
    fig2 = panel2(ds, locator, formatter, gate)

    return fig1, fig2


def panel1(ds, locator, formatter) :
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

    for axe in axes.flatten()[:-1]:
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
    ax2.set_title("DCR {:.0f} GHz Doppler Velocity".format(ds.radar_frequency*1e-9))
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

    # Wind speed / wind direction time series
    ax4b = ax4.twinx()
    wind_speed = ax4.plot(ds.time, ds["ws"].values, label = ds["ws"].attrs["long_name"], color = "red")
    ax4.set_ylabel(f"{ds['ws'].attrs['long_name']} ({ds['ws'].attrs['units']})")

    wind_direction = ax4b.plot(ds.time, ds["wd"].values, label = ds["wd"].attrs["long_name"], color = "green")
    ax4b.set_ylabel(f"{ds['wd'].attrs['long_name']} ({ds['wd'].attrs['units']})")

    ax4.grid()
    ax4.set_title("Timeseries for Wind speed and direction")
    lines = wind_speed + wind_direction
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

    def f_th(x):
        return 9.40 * (
            1 - np.exp(-1.57 * (10**3) * np.power(x * (10**-3), 1.15))
        )  # Gun and Kinzer (th.)

    def f_fit(x, a, b, c):
        return a * (1 - np.exp(-b * np.power(x * (10**-3), c)))  # target shape

    drop_density = np.nansum(
        ds["psd"].values, axis=0
    )  # sum over time dim
    psd_nonzero = np.where(drop_density != 0)
    x, y = [], []

    for k in range(
        len(psd_nonzero[0])
    ):  # add observations (size, speed) in the proportions described by the diameter/velocity distribution
        x += [ds["size_classes"][psd_nonzero[0][k]]] * int(
            drop_density[psd_nonzero[0][k], psd_nonzero[1][k]]
        )
        y += [ds["speed_classes"][psd_nonzero[1][k]]] * int(
            drop_density[psd_nonzero[0][k], psd_nonzero[1][k]]
        )

    X, Y = np.array(x), np.array(y)

    print("data ready for fitting")

    popt, pcov = so.curve_fit(f_fit, X, Y)
    y_hat = f_fit(ds["size_classes"], popt[0], popt[1], popt[2])
    y_th = f_th(ds["size_classes"])

    h = ax6.hist2d(
        X,
        Y,
        cmin=len(X) / 1000,
        bins=[ds["size_classes"], ds["speed_classes"]],
        density=False,
    )

    ax6.plot(
        ds["size_classes"],
        y_hat,
        c="green",
        label="Fit on DD measurements",
    )
    ax6.plot(
        ds["size_classes"],
        y_th,
        c="C1",
        label="Fall speed model (Gun and Kinzer)",
    )
    fig.colorbar(h[3], ax=ax6)
    ax6.legend(loc="best")
    ax6.grid()
    ax6.set_xlabel("Diameter (mm)")
    ax6.set_ylabel("Fall speed (m/s)")
    # ax6.set_xlim(disdro["size_classes"].min(), disdro["size_classes"].max())
    # ax6.set_ylim(disdro["speed_classes"].min(), disdro["speed_classes"].max())
    ax6.set_xlim(0, 5)
    ax6.set_ylim(0, 10)
    ax6.set_title("Relationship disdrometer fall speed / drop size ")

    # plt.show()

    return fig


def panel2(ds, locator, formatter, gate):
    '''Panel 2 : Zdcr data at different gates, Delta Z timeseries and distributions'''

    fig, axes = plt.subplots(2,2, figsize=(25,12))
    (ax1, ax2, ax3, ax4) = axes[0,0], axes[0,1], axes[1,0], axes[1,1]
    plt.subplots_adjust(0.06, 0.06, 0.93, 0.9, hspace=0.2, wspace=0.3)

    fig.text(
    s=pd.Timestamp(ds.time.values[0]).to_pydatetime().strftime("%Y-%m-%d") + " : Daily reflectivity data",
    fontsize=18,
    horizontalalignment="center",
    verticalalignment="center",
    y=0.97,
    x=0.5,
)
    time = ds.time.values

    for axe in axes.flatten()[:-2]:
        axe.xaxis.set_major_formatter(formatter)
        axe.set_xlabel("Time (UTC)")

    Zd = ds.Zdlog_vfov_modv_tm.sel(computed_frequencies=ds.radar_frequency.data, method="nearest").values
    plotted_alt = np.array([50, 100, 150, 300])

    # Plot Z from DCR at different gates
    for r in plotted_alt :
        Zr = ds.Zdcr.sel(range=r, method="nearest")
        r_near = Zr.range.values
        ax1.plot(time, Zr, label=f"Z from DCR @ {r_near:.0f}m", lw=0.75)
    ax1.plot(time, Zd, label = "Z modeled from DD", lw=1.5)
    ax1.grid()
    ax1.legend()
    ax1.set_ylabel("Z [dBZ]")
    ax1.set_title("Reflectivity from DCR and disdrometer")
    ax1.set_xlim(
        left=time.min(),
        right=time.max(),
    )
    ax1.set_ylim(bottom=-10)

    # Plot Delta Z DCR/DD at different gates
    for r in plotted_alt:
        dZr = ds.Zdcr.sel(range=r, method="nearest") - Zd
        r_near = dZr.range.values
        ax2.plot(time, dZr, label="$\Delta Z_{DCR/DD}$" f" @ {r_near:.0f}m", lw=1)
    ax2.grid()
    ax2.legend()
    ax2.set_ylabel("$\Delta_{Z}$ [dBZ]")
    ax2.set_title("Reflectivity differences between DCR and disdrometer")
    ax2.set_xlim(
        left=time.min(),
        right=time.max(),
    )
    ax2.set_ylim(top=10, bottom=-10)

    # Plot Delta Z pdf at a defined range (given in a configuration file)
    dZ_conf = ds.Zdcr.sel(range=gate, method="nearest") - Zd

    f = np.where((np.isfinite(dZ_conf)))[0]
    dZ_conf = dZ_conf[f]

    cdf = stats.cumfreq(dZ_conf, numbins=100)
    x = cdf.lowerlimit + np.linspace(
        0, cdf.binsize * cdf.cumcount.size, cdf.cumcount.size
    )
    ax3_cdf = ax3.twinx()
    ax3_cdf.plot(
        x,
        cdf.cumcount / len(f),
        label="$CDF Z_{DCR}$ - $Z_{Disdrometer}$",
        marker="o",
        color="green",
    )
    print(np.nanmax(dZ_conf), np.nanmin(dZ_conf))
    ax3.hist(
        dZ_conf,
        label="$Z_{{DCR}}$ - $Z_{{Disdrometer}}$, $Med = ${:.1f} $dBZ$, $\sigma = ${:.1f} $dBZ$".format(
            np.nanmedian(dZ_conf), np.nanstd(dZ_conf)
        ),
        alpha=0.4,
        color="green",
        density=True,
        bins=int(
            (
                np.nanmax(dZ_conf) - np.nanmin(dZ_conf)
            )
            / 0.5
        ),
    )
    ax3.axvline(x=np.nanmedian(dZ_conf), color="green")
    ax3.legend(loc="upper left")
    ax3.grid()
    ax3.set_xlim(left=-15, right=15)
    ax3_cdf.set_ylim(bottom=0, top=1)
    ax3.set_xlabel(r"$\Delta Z [dBZ]$")
    ax3.set_ylabel("pdf")
    ax3_cdf.set_ylabel("cdf %")
    ax3.set_title(
        "pdf of differences $Z_{{DCR}} - Z_{{DD}}$ @ {}m".format(gate)
    )

    return fig



if __name__ == "__main__" :
    MAIN_DIR = Path(__file__).parent.parent
    TEST_DIR = MAIN_DIR / "tests"
    TEST_OUT_DIR = TEST_DIR / "data/outputs"
    TEST_IN_DIR = TEST_DIR / "data/inputs"
    preprocessed_file = f"{TEST_OUT_DIR}/20210202_palaiseau_preprocessed_v1101.nc"
    config_file = f"{TEST_IN_DIR}/CONFIG_preprocessing_processing.toml"
    fig1, fig2 = daily_quicklooks(preprocessed_file, config_file)
    fig1.savefig(f"{TEST_OUT_DIR}/20210202_palaiseau_test_quicklooks_P1.png", dpi=500, transparent=False)
    fig2.savefig(f"{TEST_OUT_DIR}/20210202_palaiseau_test_quicklooks_P2.png", dpi=500, transparent=False)