import datetime as dt
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.dates import DateFormatter, HourLocator, MinuteLocator
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ccres_disdrometer_processing.metadata import INSTRUMENTS
from ccres_disdrometer_processing.plot.colormap.dcr_cmap import dcr_zh_cmap
from ccres_disdrometer_processing.plot.utils import (
    add_logo,
    get_cdf,
    get_min_max_limits,
    get_y_fit_dd,
    linear_reg_scipy,
    npdt64_to_datetime,
)

# plot parameters
lsize = 12  # label size
asize = 14  # x-y axis size
tsize = 16  # title size

lgr = logging.getLogger(__name__)


def divider(axe, size="5%", axis="off"):
    """Create space for a colorbar.

    Parameters
    ----------
    axe : matplotlib.axes
        The axe to divide.
    size : str, optional:
        Size of the divider. Defaults to "5%".
    axis : str, optional
        Should the new axis be visible. Defaults to "off".

    Returns
    -------
    matplotlib.axes
        The newly created axis.
    """
    divider = make_axes_locatable(axe)
    cax = divider.append_axes("right", size=size, pad=0.2)
    cax.axis(axis)
    return cax


def plot_preprocessed_ql_overview(
    data: xr.Dataset,
    date: dt.datetime,
    output_ql_overview: str,
    conf: dict,
    version: str,
):
    """Create overview quicklook from preprocessed data.

    Parameters
    ----------
    data : xarray.Dataset
        data read from preprocessing file.
    date : datetime.datetime
        Date of the quicklook to create
    output_ql_overview : str or pathlib.Path
        The path to the output quicklook.
    conf : dict
        The data read in the toml configuration file.
    version : str
        Version of the code.
    """
    if not isinstance(output_ql_overview, Path):
        output_ql_overview = Path(output_ql_overview)

    station = conf["location"]["STATION"]
    lat = data.attrs["geospatial_lat_max"]
    lon = data.attrs["geospatial_lon_max"]
    alt = data.attrs["geospatial_vertical_max"]

    fig, axes = plt.subplots(3, 2, figsize=(16, 10))

    # 0 - ZH reflectivity from DCR
    # ------------------------------------------------------------------------
    cmap = LinearSegmentedColormap("dcr", segmentdata=dcr_zh_cmap, N=256)
    im0 = axes[0, 0].pcolormesh(
        data.time,
        data.range,
        data.Zdcr.T,
        cmap=cmap,
        vmin=-50,
        vmax=20,
    )
    cax0 = divider(axes[0, 0], size="3%", axis="on")
    cbar0 = plt.colorbar(
        im0, cax=cax0, ax=axes[0, 0], ticks=np.arange(-50, 30, 20), extend="both"
    )
    cbar0.ax.set_ylabel(r"Zh [$dBZ$]", fontsize=lsize)
    cbar0.ax.tick_params(labelsize=lsize)
    axes[0, 0].set_ylim(0, 2500)
    axes[0, 0].set_ylabel("Altitude [m AGL]", fontsize=asize)
    axes[0, 0].set_title(
        f"DCR - {data.attrs['radar_source']}", fontsize=lsize, fontstyle="italic"
    )
    axes[0, 0].yaxis.set_major_locator(MultipleLocator(500))
    axes[0, 0].yaxis.set_minor_locator(MultipleLocator(100))

    # 1 - Air temperature & Relative Humidity from weather station
    # ------------------------------------------------------------------------
    axes[1, 0].plot(data.time, data.ta, color="#009ffd", lw=2.0, label="ta")
    _ = divider(axes[1, 0], size="3%", axis="off")
    axes[1, 0].legend(loc="upper left", fontsize=lsize)
    axes[1, 0].set_ylabel(r"Temperature [$^{o}C$]", fontsize=asize)
    axes[1, 0].set_title("weather station", fontsize=lsize, fontstyle="italic")
    axes[1, 0].yaxis.set_major_locator(MultipleLocator(2))
    axes[1, 0].yaxis.set_minor_locator(MultipleLocator(1))
    # hur
    ax12 = axes[1, 0].twinx()
    ax12.plot(data.time, data.hur, color="#ffa400", lw=2.0, label="rh")
    ax12.tick_params(labelsize=lsize)
    ax12.yaxis.set_minor_locator(MultipleLocator(1))
    ax12.yaxis.set_major_locator(MultipleLocator(5))
    ax12.legend(loc="upper right", fontsize=lsize, borderaxespad=1.0).set_zorder(2)
    ax12.set_ylabel(r"Relative Humidity [$\%$]", fontsize=asize)
    _ = divider(ax12, size="3%", axis="off")

    # 2 - precipitation from disdrometer and weather station
    # ------------------------------------------------------------------------
    axes[2, 0].plot(
        data.time,
        data.disdro_cp,
        color="#ef8a62",
        lw=2.0,
        label=data.attrs["disdrometer_source"],
    )
    axes[2, 0].plot(
        data.time, data.ams_cp, color="#999999", lw=2.0, label="Weather Station"
    )
    _ = divider(axes[2, 0], size="3%", axis="off")
    axes[2, 0].legend(loc="lower right", fontsize=lsize)
    axes[2, 0].set_ylabel(r"Cumulative rainfall [$mm$]", fontsize=asize)
    axes[2, 0].set_title(
        "Disdrometer and Weather station", fontsize=lsize, fontstyle="italic"
    )
    axes[2, 0].yaxis.set_major_locator(MultipleLocator(2))
    axes[2, 0].yaxis.set_minor_locator(MultipleLocator(1))

    # 3 - Doppler velocity from DCR
    # ------------------------------------------------------------------------
    im3 = axes[0, 1].pcolormesh(
        data.time,
        data.range,
        data.DVdcr.T,
        cmap=plt.get_cmap("coolwarm"),
        vmin=-4,
        vmax=4,
    )
    cax3 = divider(axes[0, 1], size="3%", axis="on")
    cbar3 = plt.colorbar(
        im3, cax=cax3, ax=axes[0, 1], ticks=[-4, -2, 0, 2, 4], extend="both"
    )
    cbar3.ax.set_ylabel(r"Velocity [$m.s^{-1}$]", fontsize=lsize)
    cbar3.ax.tick_params(labelsize=lsize)
    axes[0, 1].set_ylim(0, 2500)
    axes[0, 1].set_ylabel("Altitude [m AGL]", fontsize=asize)
    axes[0, 1].set_title(
        f"DCR - {data.attrs['radar_source']}", fontsize=lsize, fontstyle="italic"
    )
    axes[0, 1].yaxis.set_major_locator(MultipleLocator(500))
    axes[0, 1].yaxis.set_minor_locator(MultipleLocator(100))

    # 4 - wind speed and direction from weather station
    # ------------------------------------------------------------------------
    axes[1, 1].plot(data.time, data.ws, color="r", lw=2.0, label="Wind Speed")
    _ = divider(axes[1, 1], size="3%", axis="off")
    axes[1, 1].legend(loc="upper left", fontsize=lsize)
    axes[1, 1].set_ylabel(r"Wind Speed [$m.s^{-1}$]", fontsize=asize)
    axes[1, 1].set_title("weather station", fontsize=lsize, fontstyle="italic")
    axes[1, 1].yaxis.set_major_locator(MultipleLocator(2))
    axes[1, 1].yaxis.set_minor_locator(MultipleLocator(1))
    # wd
    ax42 = axes[1, 1].twinx()
    ax42.scatter(
        data.time,
        data.wd,
        s=10,
        color="g",
        edgecolor=None,
        label="Wind Direction",
    )
    ax42.set_ylim(0, 360)
    _ = divider(ax42, size="3%", axis="off")
    ax42.tick_params(labelsize=lsize)
    ax42.legend(loc="upper right", fontsize=lsize)
    ax42.set_ylabel(r"Wind Direction [$^{o}$]", fontsize=asize)
    ax42.yaxis.set_major_locator(MultipleLocator(60))
    ax42.yaxis.set_minor_locator(MultipleLocator(20))

    # 5 - relationship disdrometer fall speed / drop size
    # ------------------------------------------------------------------------
    y_hat, y_th, sizes_dd, classes_dd, drop_density, y_fit_ok = get_y_fit_dd(data)
    if y_fit_ok == 1:
        im5 = axes[2, 1].hist2d(
            sizes_dd,
            classes_dd,
            cmin=len(sizes_dd) / 1000,
            bins=[data["size_classes"], data["speed_classes"]],
            density=False,
        )
        axes[2, 1].plot(
            data["size_classes"],
            y_hat,
            c="green",
            lw=2,
            label=f"Fit on {data.attrs['disdrometer_source']} measurements",
        )
        axes[2, 1].plot(
            data["size_classes"],
            y_th,
            c="DarkOrange",
            lw=2,
            label="Fall speed model (Gun and Kinzer)",
        )
        axes[2, 1].legend(loc="lower right", fontsize=lsize)
        cax5 = divider(axes[2, 1], size="3%", axis="on")
        cbar5 = plt.colorbar(im5[3], cax=cax5, ax=axes[2, 1], ticks=[0, 2, 4, 6, 8, 10])
        cbar5.ax.set_ylabel(r"$\%$ of droplets total", fontsize=lsize)
        cbar5.ax.tick_params(labelsize=lsize)
    elif y_fit_ok == 0:
        axes[2, 1].text(2, 4, "No data", fontsize=asize, fontstyle="italic")
    elif y_fit_ok == -1:
        axes[2, 1].text(2, 4, "No fit", fontsize=asize, fontstyle="italic")
        cax5 = divider(axes[2, 1], size="3%", axis="off")
    axes[2, 1].set_xlabel(r"Diameter [$mm$]", fontsize=asize)
    axes[2, 1].set_ylabel(r"Fall speed [$m.s^{-1}$]", fontsize=asize)
    axes[2, 1].set_xlim(0, 5)
    axes[2, 1].set_ylim(0, 10)
    axes[2, 1].yaxis.set_major_locator(MultipleLocator(2))
    axes[2, 1].yaxis.set_minor_locator(MultipleLocator(1))
    axes[2, 1].xaxis.set_major_locator(MultipleLocator(1))
    axes[2, 1].xaxis.set_minor_locator(MultipleLocator(0.5))

    axes[2, 1].set_title(
        "Relationship disdrometer fall speed / drop size",
        fontsize=lsize,
        fontstyle="italic",
    )

    # custom
    # ------------------------------------------------------------------------
    axes[2, 0].set_xlabel("Time [UTC]", fontsize=asize)
    axes[1, 1].set_xlabel("Time [UTC]", fontsize=asize)
    for ax in axes.flatten():
        if ax in [
            axes[2, 0],
            axes[1, 1],
        ]:
            ax.tick_params(labelsize=lsize)
            ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(HourLocator(np.arange(0, 24, 3)))
            ax.xaxis.set_minor_locator(HourLocator())
            ax.set_xlim(data.time[0], data.time[-1])
        elif ax == axes[2, 1]:
            ax.tick_params(labelsize=lsize)
        else:
            ax.tick_params(labelsize=lsize, labelbottom=False)
            ax.xaxis.set_major_locator(HourLocator(np.arange(0, 24, 3)))
            ax.xaxis.set_minor_locator(HourLocator())
            ax.set_xlim(data.time[0], data.time[-1])
        ax.grid(ls="--", alpha=0.5)

    # Final layout & save / display
    # ------------------------------------------------------------------------
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    add_logo()
    fig.align_ylabels()

    fig.suptitle(
        date.strftime(
            f"Measurement site: {station} ({lat:.3f}N, {lon:.3f}E, {alt:.0f}m)\n%d-%m-%Y"  # noqa E501
        ),
        fontsize=tsize,
    )

    plt.savefig(output_ql_overview)

    # plt.show()
    plt.close()


def plot_preprocessed_ql_overview_downgraded_mode(
    data: xr.Dataset,
    date: dt.datetime,
    output_ql_overview: str,
    conf: dict,
    version: str,
):
    """Create overview quicklook from preprocessed data without meteo.

    Parameters
    ----------
    data : xarray.Dataset
        Data read from the preprocessed file.
    date : datetime.datetime
        Date of the data.
    output_ql_overview : str or pathlib.Path
        The path to the output quicklook to create.
    conf : dict
        The data read in the toml configuration file.
    version : str
        Version of the code.
    """
    if not isinstance(output_ql_overview, Path):
        output_ql_overview = Path(output_ql_overview)

    station = conf["location"]["STATION"]
    lat = data.attrs["geospatial_lat_max"]
    lon = data.attrs["geospatial_lon_max"]
    alt = data.attrs["geospatial_vertical_max"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 0 - ZH reflectivity from DCR
    # ------------------------------------------------------------------------
    cmap = LinearSegmentedColormap("dcr", segmentdata=dcr_zh_cmap, N=256)
    im0 = axes[0, 0].pcolormesh(
        data.time,
        data.range,
        data.Zdcr.T,
        cmap=cmap,
        vmin=-50,
        vmax=20,
    )
    cax0 = divider(axes[0, 0], size="3%", axis="on")
    cbar0 = plt.colorbar(
        im0, cax=cax0, ax=axes[0, 0], ticks=np.arange(-50, 30, 20), extend="both"
    )
    cbar0.ax.set_ylabel(r"Zh [$dBZ$]", fontsize=lsize)
    cbar0.ax.tick_params(labelsize=lsize)
    axes[0, 0].set_ylim(0, 2500)
    axes[0, 0].set_ylabel("Altitude [m AGL]", fontsize=asize)
    axes[0, 0].set_title(
        f"DCR - {data.attrs['radar_source']}", fontsize=lsize, fontstyle="italic"
    )
    axes[0, 0].yaxis.set_major_locator(MultipleLocator(500))
    axes[0, 0].yaxis.set_minor_locator(MultipleLocator(100))

    # 1 - precipitation from disdrometer and weather station
    # ------------------------------------------------------------------------
    axes[1, 0].plot(
        data.time,
        data.disdro_cp,
        color="#ef8a62",
        lw=2.0,
        label=data.attrs["disdrometer_source"],
    )
    _ = divider(axes[1, 0], size="3%", axis="off")
    axes[1, 0].legend(loc="lower right", fontsize=lsize)
    axes[1, 0].set_ylabel(r"Cumulative rainfall [$mm$]", fontsize=asize)
    axes[1, 0].set_title(
        "Disdrometer and Weather station", fontsize=lsize, fontstyle="italic"
    )
    axes[1, 0].yaxis.set_major_locator(MultipleLocator(2))
    axes[1, 0].yaxis.set_minor_locator(MultipleLocator(1))

    # 2 - Doppler velocity from DCR
    # ------------------------------------------------------------------------
    im2 = axes[0, 1].pcolormesh(
        data.time,
        data.range,
        data.DVdcr.T,
        cmap=plt.get_cmap("coolwarm"),
        vmin=-4,
        vmax=4,
    )
    cax2 = divider(axes[0, 1], size="3%", axis="on")
    cbar2 = plt.colorbar(
        im2, cax=cax2, ax=axes[0, 1], ticks=[-4, -2, 0, 2, 4], extend="both"
    )
    cbar2.ax.set_ylabel(r"Velocity [$m.s^{-1}$]", fontsize=lsize)
    cbar2.ax.tick_params(labelsize=lsize)
    axes[0, 1].set_ylim(0, 2500)
    axes[0, 1].set_ylabel("Altitude [m AGL]", fontsize=asize)
    axes[0, 1].set_title(
        f"DCR - {data.attrs['radar_source']}", fontsize=lsize, fontstyle="italic"
    )
    axes[0, 1].yaxis.set_major_locator(MultipleLocator(500))
    axes[0, 1].yaxis.set_minor_locator(MultipleLocator(100))

    # 3 - relationship disdrometer fall speed / drop size
    # ------------------------------------------------------------------------
    y_hat, y_th, sizes_dd, classes_dd, drop_density, y_fit_ok = get_y_fit_dd(data)
    if y_fit_ok == 1:
        im3 = axes[1, 1].hist2d(
            sizes_dd,
            classes_dd,
            cmin=len(sizes_dd) / 1000,
            bins=[data["size_classes"], data["speed_classes"]],
            density=False,
        )

        axes[1, 1].plot(
            data["size_classes"],
            y_hat,
            c="green",
            lw=2,
            label=f"Fit on {data.attrs['disdrometer_source']} measurements",
        )
        axes[1, 1].plot(
            data["size_classes"],
            y_th,
            c="DarkOrange",
            lw=2,
            label="Fall speed model (Gun and Kinzer)",
        )
        axes[1, 1].legend(loc="lower right", fontsize=lsize)
        cax3 = divider(axes[1, 1], size="3%", axis="on")
        cbar3 = plt.colorbar(im3[3], cax=cax3, ax=axes[1, 1], ticks=[0, 2, 4, 6, 8, 10])
        cbar3.ax.set_ylabel(r"$\%$ of droplets total", fontsize=lsize)
        cbar3.ax.tick_params(labelsize=lsize)
    elif y_fit_ok == 0:
        axes[1, 1].text(2, 4, "No data", fontsize=asize, fontstyle="italic")
    elif y_fit_ok == -1:
        axes[1, 1].text(2, 4, "No fit", fontsize=asize, fontstyle="italic")
        cax3 = divider(axes[1, 1], size="3%", axis="off")
    axes[1, 1].set_xlabel(r"Diameter [$mm$]", fontsize=asize)
    axes[1, 1].set_ylabel(r"Fall speed [$m.s^{-1}$]", fontsize=asize)
    axes[1, 1].set_xlim(0, 5)
    axes[1, 1].set_ylim(0, 10)
    axes[1, 1].yaxis.set_major_locator(MultipleLocator(2))
    axes[1, 1].yaxis.set_minor_locator(MultipleLocator(1))
    axes[1, 1].xaxis.set_major_locator(MultipleLocator(1))
    axes[1, 1].xaxis.set_minor_locator(MultipleLocator(0.5))

    axes[1, 1].set_title(
        "Relationship disdrometer fall speed / drop size",
        fontsize=lsize,
        fontstyle="italic",
    )

    # custom
    # ------------------------------------------------------------------------
    axes[1, 0].set_xlabel("Time [UTC]", fontsize=asize)
    axes[0, 1].set_xlabel("Time [UTC]", fontsize=asize)
    for ax in axes.flatten():
        if ax != axes[1, 1]:
            ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(HourLocator(np.arange(0, 24, 3)))
            ax.xaxis.set_minor_locator(HourLocator())
            ax.set_xlim(data.time[0], data.time[-1])
        ax.tick_params(labelsize=lsize)
        ax.grid(ls="--", alpha=0.5)

    # Final layout & save / display
    # ------------------------------------------------------------------------
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    add_logo()
    fig.align_ylabels()

    fig.suptitle(
        date.strftime(
            f"Measurement site: {station} ({lat:.3f}N, {lon:.3f}E, {alt:.0f}m)\n{station}\n%d-%m-%Y"  # noqa E501
        ),
        fontsize=tsize,
    )

    plt.savefig(output_ql_overview)

    # plt.show()
    plt.close()


def plot_preprocessed_ql_overview_zh(
    data: xr.Dataset,
    date: dt.datetime,
    output_ql_overview_zh: str,
    conf: dict,
    version: str,
):
    """Create overview quicklook of reflectivity.

    Parameters
    ----------
    data : xarray.Dataset
        Data read from preprocess file.
    date : datetime.datetime
        Date of the quicklook to create.
    output_ql_overview_zh : str or pathlib.Path
        The path to the output quicklook.
    conf : dict
        The data read in the toml configuration file.
    version : str
        Version of the code.
    """
    site = data.attrs["location"]
    station = conf["location"]["STATION"]
    lat = data.attrs["geospatial_lat_max"]
    lon = data.attrs["geospatial_lon_max"]
    alt = data.attrs["geospatial_vertical_max"]
    selected_alt = conf["instrument_parameters"]["DCR_DZ_RANGE"]

    fig, axes = plt.subplot_mosaic(
        [["top", "top"], ["left", "right"]], constrained_layout=True, figsize=(16, 10)
    )

    # 0 - Zh from DCR and disdrometer
    # ------------------------------------------------------------------------
    ZH_DD = data["Zdlog_vfov_modv_tm"].sel(
        radar_frequencies=data["radar_frequency"].values, method="nearest"
    )
    ZH_DCR = data["Zdcr"].sel(range=selected_alt, method="nearest")
    axes["top"].plot(
        data.time,
        ZH_DCR,
        label=r"$Z_{H}$" + f" from DCR @ {ZH_DCR.range.values:.0f}m",
        lw=2,
    )
    axes["top"].plot(
        data.time, ZH_DD, color="k", label=r"$Z_{H}$ modeled from DD", lw=2
    )
    axes["top"].legend(loc="upper right", fontsize=lsize)
    axes["top"].set_ylabel(r"$Z_{H}$ [$dBZ$]", fontsize=asize)
    axes["top"].set_title(
        f"Reflectivity from {data.attrs['radar_source']} DCR and {data.attrs['disdrometer_source']} disdrometer",  # noqa E501
        fontsize=lsize,
        fontstyle="italic",
    )
    axes["top"].set_ylim(-60, 30)
    axes["top"].yaxis.set_major_locator(MultipleLocator(10))
    axes["top"].yaxis.set_minor_locator(MultipleLocator(2))
    axes["top"].set_xlim(data.time[0], data.time[-1])
    axes["top"].xaxis.set_major_formatter(DateFormatter("%H:%M"))
    axes["top"].xaxis.set_major_locator(HourLocator(np.arange(0, 24, 3)))
    axes["top"].xaxis.set_minor_locator(HourLocator())

    # 1 - PDF of Zh differences between DCR and disdrometer
    # ------------------------------------------------------------------------
    # Plot Delta Z pdf at a defined range (given in a configuration file)
    ZH_gate = data["Zdcr"].sel(range=selected_alt, method="nearest")
    true_alt_zh_gate = ZH_gate.range.values
    delta_ZH = ZH_gate.values - ZH_DD.values
    ind = np.where(np.isfinite(delta_ZH))[0]
    delta_ZH = delta_ZH[ind]
    bin_width = 0.5

    # histogram
    if delta_ZH.size != 0:
        delta_zh_median = np.round(np.nanmedian(delta_ZH), 2)
        delta_zh_std = np.round(np.nanstd(delta_ZH), 2)
        xmin = int(delta_zh_median - 2 * delta_zh_std)
        xmax = int(delta_zh_median + 2 * delta_zh_std)
        axes["left"].hist(
            delta_ZH,
            bins=int((np.nanmax(delta_ZH) - np.nanmin(delta_ZH)) / bin_width),
            weights=(np.ones(delta_ZH.size) / delta_ZH.size) * 100,
            color="green",
            label=r"$Z_{H}^{DCR}$ - $Z_{H}^{DD}$, $Med = $"
            + str(delta_zh_median)
            + " $dBZ$, $\sigma = $"
            + str(delta_zh_std)
            + " $dBZ$",
            alpha=0.4,
        )
        axes["left"].axvline(x=np.nanmedian(delta_ZH), color="green")
        axes["left"].axvline(x=0, color="k", ls="--")
        axes["left"].legend(loc="upper left", fontsize=lsize)
        axes["left"].set_yticks(axes["left"].get_yticks())
        axes["left"].set_yticklabels(np.round(bin_width * axes["left"].get_yticks(), 2))
        axes["left"].set_xlim(xmin, xmax)
    else:
        axes["left"].text(0, 0.5, "No data", fontsize=asize, fontstyle="italic")
        axes["left"].set_xlim(0, 1)
    axes["left"].set_xlabel(r"$\Delta$ $Z_{H}$ [$dBZ$]", fontsize=asize)
    axes["left"].set_ylabel(r"Rel. Occ [$\%$]", fontsize=asize)
    axes["left"].yaxis.set_major_locator(MultipleLocator(2))
    axes["left"].yaxis.set_minor_locator(MultipleLocator(0.5))
    axes["left"].xaxis.set_major_locator(MultipleLocator(5))
    axes["left"].xaxis.set_minor_locator(MultipleLocator(1))

    dcr_source = data.attrs["radar_source"]
    dd_source = data.attrs["disdrometer_source"]
    axes["left"].set_title(
        r"PDF of differences : $Z_{H}$ from "
        + dcr_source
        + " @ "
        + f"{ZH_DCR.range.values:.0f}m"
        + "m - "
        + r"$Z_{H}$"
        + " from "
        + dd_source,
        fontsize=lsize,
        fontstyle="italic",
    )
    # CDF
    if delta_ZH.size != 0:
        cdf, x_ = get_cdf(delta_ZH, nbins=100)
        ax12 = axes["left"].twinx()
        ax12.plot(
            x_,
            (cdf.cumcount / len(ind)) * 100,
            label="$CDF Z_{H}^{DCR}$ - $Z_{H}^{Disdrometer}$",
            marker="o",
            color="green",
            lw=2,
        )
        ax12.set_ylabel(r"Cum. Occ. [$\%$]", fontsize=asize)
        ax12.set_ylim(0, 100)
        ax12.tick_params(labelsize=lsize)
        ax12.yaxis.set_major_locator(MultipleLocator(20))
        ax12.yaxis.set_minor_locator(MultipleLocator(5))

    # 2 - Zh_DCR vs Zh_Disdrometer
    # ------------------------------------------------------------------------
    if delta_ZH.size != 0:
        ax_min, ax_max = get_min_max_limits(ZH_DD, ZH_gate)
        axes["right"].scatter(
            ZH_DD,
            ZH_gate,
            s=30,
            edgecolor=None,
            label=r"$Z_{H}^{DCR}$ @ "
            + f"{true_alt_zh_gate:.0f}"
            + r"m vs $Z_{H}^{DD}$",
        )
        axes["right"].plot([ax_min, ax_max], [ax_min, ax_max], color="k", label="x=y")
        axes["right"].set_xlim(ax_min, ax_max)
        axes["right"].set_ylim(ax_min, ax_max)
    axes["right"].set_aspect("equal", anchor="C")
    axes["right"].set_xlabel(
        f"{INSTRUMENTS['disdrometer'][data.attrs['disdrometer_source']]}"
        + r" DD reflectivity [$dBZ$]",
        fontsize=asize,
    )
    axes["right"].set_ylabel(
        f"{INSTRUMENTS['dcr'][data.attrs['radar_source']]}"
        + r" DCR reflectivity [$dBZ$]",
        fontsize=asize,
    )
    axes["right"].legend(fontsize=lsize)
    axes["right"].yaxis.set_major_locator(MultipleLocator(10))
    axes["right"].yaxis.set_minor_locator(MultipleLocator(5))
    axes["right"].xaxis.set_major_locator(MultipleLocator(10))
    axes["right"].xaxis.set_minor_locator(MultipleLocator(5))

    # custom
    # ------------------------------------------------------------------------
    for pos in list(axes):
        axes[pos].grid(ls="--", alpha=0.5)
        axes[pos].tick_params(labelsize=lsize)

    # Final layout & save / display
    # ------------------------------------------------------------------------
    plt.tight_layout()
    plt.subplots_adjust(top=0.86, hspace=0.2, wspace=0.15)
    add_logo()
    fig.align_ylabels()

    fig.suptitle(
        date.strftime(
            f"Measurement site: {site} ({lat:.3f}N, {lon:.3f}E, {alt:.0f}m)\n{station}\n%d-%m-%Y"  # noqa E501
        ),
        fontsize=tsize,
    )

    plt.savefig(output_ql_overview_zh)

    # plt.show()
    plt.close()


def plot_processed_ql_summary(
    ds_pro: xr.Dataset, mask_output_ql_summary: str, conf: dict, version: str
):
    """Create summary quicklook from processed data.

    Parameters
    ----------
    ds_pro : xarray.Dataset
        data read from processing file.
    mask_output_ql_summary : str or pathlib.Path
        The mask of the path to the output quicklook.
    conf : dict
        The data read in the toml configuration file.
    version : str
        Version of the code.
    """
    selected_alt = conf["instrument_parameters"]["DCR_DZ_RANGE"]

    if ds_pro.events.size != 0:
        for n, event in enumerate(ds_pro["events"]):  # noqa B007
            subdata = ds_pro.sel(
                time=slice(
                    ds_pro["start_event"][n].values, ds_pro["end_event"][n].values
                )
            )
            # =================
            # PLOT
            # =================
            fig, axes = plt.subplots(1, 3, figsize=(16, 6))
            # 0 - PDF
            # ------------------------------------------------------------
            ax02 = axes[0].twinx()
            mask_valid = np.isfinite(
                subdata["Delta_Z"].sel(range=selected_alt, method="nearest")
            ) & (subdata["QC_overall"] == 1)
            delta_Z = (
                subdata["Delta_Z"]
                .sel(range=selected_alt, method="nearest")[mask_valid]
                .values
            )
            if np.all(np.isnan(delta_Z)):
                continue
            vlim = np.ceil(np.max(np.abs([delta_Z.min(), delta_Z.max()])))
            bins = np.arange(-vlim, vlim + 0.1, 1)
            axes[0].hist(
                delta_Z,
                bins=bins,
                weights=(np.ones(delta_Z.size) / delta_Z.size) * 100,
                label=f"DCR range gate at {selected_alt:.0f}m",
            )
            sigma = np.nanstd(delta_Z)
            ymin, ymax = axes[0].get_ylim()
            axes[0].vlines(0, 0, ymax, color="k", ls="-")
            axes[0].vlines(
                -sigma, 0, ymax, ls="--", color="r", label=r"$\pm$1$\sigma_{std}$"
            )
            axes[0].vlines(sigma, 0, ymax, ls="--", color="r")
            #
            axes[0].yaxis.set_major_locator(MultipleLocator(5))
            axes[0].yaxis.set_minor_locator(MultipleLocator(1))
            axes[0].xaxis.set_major_locator(MultipleLocator(3))
            axes[0].xaxis.set_minor_locator(MultipleLocator(1))
            axes[0].set_ylabel("Density [%]", fontsize=asize)
            axes[0].set_xlabel(r"$\Delta$$Z_{DCR}^{DD}$ [dBZ]", fontsize=asize)
            axes[0].legend(loc="upper left", fontsize=lsize)
            #
            hist, bin_edges = np.histogram(
                delta_Z,
                bins=bins,
                weights=(np.ones(delta_Z.size) / delta_Z.size) * 100,
            )
            ax02.plot(bin_edges[:-1], np.cumsum(hist), marker=".", color="b", lw=2)
            ax02.set_ylabel("CDF [%]", fontsize=lsize)
            ax02.set_ylim(0, 101)
            ax02.yaxis.set_major_locator(MultipleLocator(20))
            ax02.yaxis.set_minor_locator(MultipleLocator(5))
            ax02.tick_params(labelsize=lsize)

            # 1 - scatter & fit
            # ------------------------------------------------------------
            mask_Z_valid = (
                np.isfinite(subdata["Zdcr"].sel(range=selected_alt, method="nearest"))
                & np.isfinite(subdata["Zdd"])
                & np.isfinite(
                    subdata["Delta_Z"].sel(range=selected_alt, method="nearest")
                )
                & (subdata["QC_overall"] == 1)
            )
            mask_Z_false = (
                np.isfinite(subdata["Zdcr"].sel(range=selected_alt, method="nearest"))
                & np.isfinite(subdata["Zdd"])
                & np.isfinite(
                    subdata["Delta_Z"].sel(range=selected_alt, method="nearest")
                )
                & (subdata["QC_overall"] == 0)
            )

            Z_dcr_valid = (
                subdata["Zdcr"]
                .sel(range=selected_alt, method="nearest")[mask_Z_valid]
                .values
            )
            Z_dd_valid = subdata["Zdd"][mask_Z_valid].values
            Z_dcr_false = (
                subdata["Zdcr"]
                .sel(range=selected_alt, method="nearest")[mask_Z_false]
                .values
            )
            Z_dd_false = subdata["Zdd"][mask_Z_false].values

            axes[1].scatter(Z_dd_valid, Z_dcr_valid, s=30, edgecolor=None)
            axes[1].scatter(
                Z_dd_false,
                Z_dcr_false,
                s=30,
                color="LightGray",
                alpha=0.5,
                edgecolor=None,
            )
            # Linear regression fit
            slope, intercept, r_value, p_value, std_err = linear_reg_scipy(
                Z_dd_valid, Z_dcr_valid
            )
            fitted_curve = slope * Z_dd_valid + intercept
            fitted_curve_txt = f"y = {slope:.2f}x {intercept:+.2f}"
            axes[1].plot(
                Z_dd_valid, fitted_curve, color="r", lw=2, label=fitted_curve_txt
            )
            #
            vars_minmax = [
                Z_dcr_valid,
                Z_dd_valid,
                Z_dcr_false,
                Z_dd_false,
            ]

            zmin, zmax = (
                np.nanmin([np.nanmin(var) for var in vars_minmax if var.size != 0]),
                np.nanmax([np.nanmax(var) for var in vars_minmax if var.size != 0]),
            )
            vmin, vmax = np.floor(zmin / 5) * 5, np.ceil(zmax / 5) * 5
            axes[1].plot([vmin, vmax], [vmin, vmax], "k", label=r"$Z_{DD}$ = $Z_{DCR}$")
            axes[1].legend(loc="upper left", fontsize=lsize)
            axes[1].set_xlim(vmin, vmax)
            axes[1].set_ylim(vmin, vmax)
            axes[1].yaxis.set_major_locator(MultipleLocator(5))
            axes[1].yaxis.set_minor_locator(MultipleLocator(1))
            axes[1].xaxis.set_major_locator(MultipleLocator(5))
            axes[1].xaxis.set_minor_locator(MultipleLocator(1))
            axes[1].set_ylabel(r"$Z_{DCR}$ [dBZ]", fontsize=asize)
            axes[1].set_xlabel(r"$Z_{DD}$ [dBZ]", fontsize=asize)
            axes[1].set_title(f"n = {len(Z_dcr_valid)} points", fontsize=asize)

            # 2 - Metrics
            # ------------------------------------------------------------
            axes[2].set_ylim(0, 2)
            axes[2].axis("off")
            axes[2].annotate(
                f"Event duration : {int(ds_pro['event_length'][n].values)} minutes",
                (0, 1.5),
                fontsize=asize,
            )
            axes[2].annotate(
                f"Rainfall accumulation : {ds_pro['rain_accumulation'][n].values:.2f}mm",  # noqa E501
                (0, 1),
                fontsize=asize,
            )
            # TODO: limits to be defined with CCRES !
            if (np.nanmean(delta_Z) >= -2) & (np.nanmean(delta_Z) <= 2):
                color = "g"
            else:
                color = "r"
            axes[2].annotate(
                r"Mean $\Delta$$Z_{DCR}^{DD}$" + f" : {np.nanmean(delta_Z):.2f}dBZ",
                (0, 0.5),
                fontsize=tsize,
                fontweight="bold",
                color=color,
            )  # TODO: BE CAREFULL WITH mean and reflectivity ! dB !!!

            # custom
            # ------------------------------------------------------------
            for k, ax in enumerate(axes.flatten()):
                if k in [0, 1]:
                    ax.grid(ls="--", alpha=0.5)
                    ax.tick_params(labelsize=lsize)
            # suptitle
            start_event = npdt64_to_datetime(ds_pro["start_event"][n].values).strftime(  # noqa
                "%H:%M %d-%m-%Y"
            )
            end_event = npdt64_to_datetime(ds_pro["end_event"][n].values).strftime(  # noqa
                "%H:%M %d-%m-%Y"
            )
            event_number = n + 1
            suptitle = (
                f"CCRES PROCESSING @ {ds_pro.station_name}\n"
                f"File processing on {np.datetime_as_string(ds_pro.time[0].values, unit='D')}\n"  # noqa
                f"Event {event_number} from {start_event} to {end_event}"
            )
            fig.suptitle(
                suptitle,
                fontsize=tsize,
            )
            #
            add_logo()
            plt.tight_layout()

            output_png = mask_output_ql_summary.format(n)
            plt.savefig(output_png)  # noqa
            plt.close()
    else:
        lgr.info("No event to plot")


def plot_processed_ql_detailled(
    ds_pro: xr.Dataset,
    ds_prepro: xr.Dataset,
    mask_output_ql_detailled: str,
    conf: dict,
    version: str,
):
    """Create detailled quicklook from processed data.

    Parameters
    ----------
    ds_pro : xarray.Dataset
        data read from processing file.
    ds_prepro : xarray.Dataset
        data read from preprocessing file.
    mask_output_ql_detailled : str or pathlib.Path
        The mask of the path to the output quicklook.
    conf : dict
        The data read in the toml configuration file.
    version : str
        Version of the code.
    """
    # TODO: properly
    selected_alt = conf["instrument_parameters"]["DCR_DZ_RANGE"]

    if ds_pro.events.size != 0:
        for n, event in enumerate(ds_pro["events"]):  # noqa B007
            # focus on each rain event
            date_start = npdt64_to_datetime(
                ds_pro["start_event"][n].values
            ) - dt.timedelta(hours=1)
            date_end = npdt64_to_datetime(ds_pro["end_event"][n].values) + dt.timedelta(
                hours=1
            )
            subdata_pro = ds_pro.sel(time=slice(date_start, date_end))
            subdata_prepro = ds_prepro.sel(time=slice(date_start, date_end))

            # ==============
            # PLOT
            # ==============
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(15, 16)
            ax1 = fig.add_subplot(gs[:4, 3:7])
            ax2 = fig.add_subplot(gs[:4, 9:13])
            ax3 = fig.add_subplot(gs[5:7, :])
            ax4 = fig.add_subplot(gs[7:9, :])
            ax5 = fig.add_subplot(gs[9:11, :])
            ax6 = fig.add_subplot(gs[11:13, :])
            ax7 = fig.add_subplot(gs[13:, :])
            axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

            # 1 - PDF
            # ------------------------------------------------------------
            ax12 = axes[0].twinx()
            mask = np.isfinite(
                subdata_pro["Delta_Z"].sel(range=selected_alt, method="nearest")
            )
            delta_Z = (
                subdata_pro["Delta_Z"]
                .sel(range=selected_alt, method="nearest")[mask]
                .values
            )
            if np.all(np.isnan(delta_Z)):
                continue
            vlim = np.ceil(np.max(np.abs([delta_Z.min(), delta_Z.max()])))
            bins = np.arange(-vlim, vlim + 0.1, 1)
            ax1.hist(
                delta_Z,
                bins=bins,
                weights=(np.ones(delta_Z.size) / delta_Z.size) * 100,
                label=f"range gate at {selected_alt:.0f}m",
            )
            ax1.yaxis.set_major_locator(MultipleLocator(5))
            ax1.yaxis.set_minor_locator(MultipleLocator(1))
            ax1.xaxis.set_major_locator(MultipleLocator(3))
            ax1.xaxis.set_minor_locator(MultipleLocator(1))
            ax1.set_ylabel("Density [%]", fontsize=asize)
            ax1.set_xlabel(r"$\Delta$$Z_{DCR}^{DD}$ [dBZ]", fontsize=asize)
            #
            sigma = np.nanstd(delta_Z)
            ymin, ymax = ax1.get_ylim()
            ax1.vlines(0, 0, ymax, color="k", ls="-")
            ax1.vlines(
                -sigma, 0, ymax, ls="--", color="r", label=r"$\pm$1$\sigma_{std}$"
            )
            ax1.vlines(sigma, 0, ymax, ls="--", color="r")
            ax1.legend(loc="upper left", fontsize=lsize)

            #
            hist, bin_edges = np.histogram(
                delta_Z,
                bins=bins,
                weights=(np.ones(delta_Z.size) / delta_Z.size) * 100,
            )
            ax12.plot(bin_edges[:-1], np.cumsum(hist), marker=".", color="b", lw=2)
            ax12.set_ylabel("CDF [%]", fontsize=lsize)
            ax12.set_ylim(0, 100)
            ax12.yaxis.set_major_locator(MultipleLocator(20))
            ax12.yaxis.set_minor_locator(MultipleLocator(5))
            ax12.tick_params(labelsize=lsize)

            # 2 - scatter & fit
            # ------------------------------------------------------------
            mask_Z = np.isfinite(
                subdata_pro["Zdcr"].sel(range=selected_alt, method="nearest")
            ) & np.isfinite(subdata_pro["Zdd"])
            Z_dcr = (
                subdata_pro["Zdcr"]
                .sel(range=selected_alt, method="nearest")[mask_Z]
                .values
            )
            Z_dd = subdata_pro["Zdd"][mask_Z].values
            ax2.scatter(Z_dd, Z_dcr, s=30, edgecolor=None)
            zmin, zmax = (
                np.nanmin([Z_dcr.min(), Z_dd.min()]),
                np.nanmax([Z_dcr.max(), Z_dd.max()]),
            )
            vmin, vmax = round(zmin / 5) * 5, round(zmax / 5) * 5
            ax2.plot([vmin, vmax], [vmin, vmax], "k", label=r"$Z_{DD}$ = $Z_{DCR}$")
            ax2.legend(loc="upper left", fontsize=lsize)
            ax2.set_xlim(vmin, vmax)
            ax2.set_ylim(vmin, vmax)
            ax2.yaxis.set_major_locator(MultipleLocator(5))
            ax2.yaxis.set_minor_locator(MultipleLocator(1))
            ax2.xaxis.set_major_locator(MultipleLocator(5))
            ax2.xaxis.set_minor_locator(MultipleLocator(1))
            ax2.set_ylabel(r"$Z_{DCR}$ [dBZ]", fontsize=asize)
            ax2.set_xlabel(r"$Z_{DD}$ [dBZ]", fontsize=asize)
            ax2.set_title(f"n = {len(Z_dcr)} points", fontsize=asize)
            divider(ax2, axis="off")

            # 3 - DCR
            # ------------------------------------------------------------
            cax3 = divider(ax3, size="2%", axis="on")
            im3 = ax3.pcolormesh(
                subdata_prepro.time,
                subdata_prepro.range / 1000,
                subdata_prepro.Zdcr.T,
                vmin=-50,
                vmax=20,
            )
            ax3.yaxis.set_major_locator(MultipleLocator(1))
            ax3.yaxis.set_minor_locator(MultipleLocator(0.2))
            ax3.set_ylabel("Altitude\n[km]", fontsize=asize)
            cbar3 = plt.colorbar(
                im3, cax=cax3, ax=ax3, ticks=np.arange(-50, 30, 20), extend="both"
            )
            cbar3.ax.set_ylabel(r"Zh [$dBZ$]", fontsize=lsize)
            cbar3.ax.tick_params(labelsize=lsize)

            # 4 - Zdcr, Zdd
            # ------------------------------------------------------------
            ax4.plot(
                subdata_pro.time,
                subdata_pro.Zdcr.sel(range=selected_alt, method="nearest"),
                lw=2,
                color="g",
                label="$Z_{DCR}$",
            )
            ax4.plot(
                subdata_pro.time,
                subdata_pro.Zdd,
                lw=2,
                color="DarkOrange",
                label=r"$Z_{DD}$",
            )
            ax4.set_ylabel("Zh\n[dBZ]", fontsize=asize)
            ax4.legend(fontsize=lsize)
            ax4.yaxis.set_major_locator(MultipleLocator(20))
            ax4.yaxis.set_minor_locator(MultipleLocator(5))
            divider(ax4, axis="off")

            # 5 - Delta Zdcr - Zdd
            # ------------------------------------------------------------
            ax5.scatter(
                subdata_pro.time,
                subdata_pro.Delta_Z.sel(range=selected_alt, method="nearest"),
                s=30,
                color="DimGray",
            )
            ax5.hlines(0, date_start, date_end, color="k", lw=1)
            ax5.set_ylabel(r"$\Delta$" + "Zh\n[dBZ]", fontsize=asize)
            ax5.yaxis.set_major_locator(MultipleLocator(5))
            ax5.yaxis.set_minor_locator(MultipleLocator(1))
            divider(ax5, axis="off")

            # 6 - Rainfall acc
            # ------------------------------------------------------------
            ax6.fill_between(
                x=subdata_pro.time,
                y1=subdata_pro.disdro_cp_since_event_begin,
            )
            ax6.hlines(3, date_start, date_end, lw=1, color="k")
            ax6.set_ylabel("DD acc.\n[mm]", fontsize=asize)
            ax6.yaxis.set_major_locator(MultipleLocator(3))
            ax6.yaxis.set_minor_locator(MultipleLocator(1))
            divider(ax6, axis="off")

            # 7 - QC
            # ------------------------------------------------------------
            qc_colors = ["r", "g"]
            for j, qc in enumerate(
                ["QC_pr", "QC_vdsd_t", "QF_rainfall_amount", "QC_overall"]
            ):
                if j != 3:
                    for level in [0, 1]:
                        mask = subdata_pro[qc] == level
                        values = subdata_pro.time[mask]
                        if values.size != 0:
                            ax7.bar(
                                values,
                                height=np.ones(values.size),
                                bottom=j,
                                width=1 / 1440,
                                color=qc_colors[level],
                                alpha=0.5,
                            )
                else:
                    for level in [0, 1]:
                        mask = subdata_pro[qc] == level
                        values = subdata_pro.time[mask]
                        if values.size != 0:
                            ax7.bar(
                                values,
                                height=np.ones(values.size),
                                bottom=j,
                                width=1 / 1440,
                                color=qc_colors[level],
                            )

            ax7.set_ylim(0.0, 4)
            ax7.set_yticks([0.5, 1.5, 2.5, 3.5])
            ax7.set_yticklabels(
                ["QC_pr", "QC_vdsd_t", "QF_rainfall_amount", "QC_overall"]
            )
            for line in [1, 2]:
                ax7.hlines(line, date_start, date_end, color="k", lw=1)
            ax7.set_xlabel("Time [UTC]", fontsize=asize)
            divider(ax7, axis="off")

            # custom
            # ------------------------------------------------------------
            for ax in axes:
                if ax in [ax3, ax4, ax5, ax6]:
                    ax.tick_params(labelbottom=False)
                if ax not in [ax1, ax2]:
                    ax.set_xlim(date_start, date_end)
                    ax.xaxis.set_major_locator(HourLocator(interval=3))
                    ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
                    ax.xaxis.set_minor_locator(MinuteLocator([0, 15, 30, 45]))
                if ax != ax7:
                    ax.grid(ls="--", alpha=0.5)
                ax.tick_params(labelsize=lsize)

            # suptitle
            start_event = npdt64_to_datetime(ds_pro["start_event"][n].values).strftime(  # noqa
                "%H:%M %d-%m-%Y"
            )
            end_event = npdt64_to_datetime(ds_pro["end_event"][n].values).strftime(  # noqa
                "%H:%M %d-%m-%Y"
            )
            event_number = n + 1
            suptitle = (
                f"CCRES PROCESSING @ {ds_pro.station_name}\n"
                f"File processing on {np.datetime_as_string(ds_pro.time[0].values, unit='D')}\n"  # noqa
                f"Event {event_number} from {start_event} to {end_event}"
            )
            fig.suptitle(
                suptitle,
                fontsize=tsize,
            )
            #
            add_logo()
            plt.tight_layout()
            fig.align_ylabels()
            plt.subplots_adjust(
                left=0.15, bottom=0.05, right=0.95, top=0.85, hspace=0.9
            )

            output_png = mask_output_ql_detailled.format(n)
            plt.savefig(output_png)
            plt.close()
    else:
        lgr.info("No event to plot")
