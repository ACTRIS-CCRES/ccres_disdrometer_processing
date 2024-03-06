import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.dates import DateFormatter, HourLocator
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ccres_disdrometer_processing.cli.assets.colormap.dcr_cmap import dcr_zh_cmap
from ccres_disdrometer_processing.cli.plot.utils import (
    add_logo,
    get_cdf,
    get_min_max_limits,
    get_y_fit_dd,
)


def divider(axe, size="5%", axis="off"):
    """Create space for a colorbar.
    Args:
        axe (_type_): _description_
        size (str, optional): _description_. Defaults to "5%".
        axis (str, optional): _description_. Defaults to "off".

    Returns:
        _type_: _description_
    """
    divider = make_axes_locatable(axe)
    cax = divider.append_axes("right", size=size, pad=0.2)
    cax.axis(axis)
    return cax


def plot_ql_overview(
    data: xr.Dataset,
    date: dt.datetime,
    output_ql_overview: str,
    conf: object,
    params: object,
    version: str,
):
    """_summary_

    Args:
        data (xr.Dataset): _description_
        date (dt.datetime): _description_
        output_ql_overview (str): _description_
        conf (object): _description_
        params (object): _description_
        version (str): _description_
    """

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
    cbar0.ax.set_ylabel(r"Zh [$dBZ$]", fontsize=params.lsize)
    cbar0.ax.tick_params(labelsize=params.lsize)
    axes[0, 0].set_ylim(0, 2500)
    axes[0, 0].set_ylabel("Altitude [m AGL]", fontsize=params.asize)
    axes[0, 0].set_title(
        f"DCR - {data.attrs['radar_source']}", fontsize=params.lsize, fontstyle="italic"
    )
    axes[0, 0].yaxis.set_major_locator(MultipleLocator(500))
    axes[0, 0].yaxis.set_minor_locator(MultipleLocator(100))

    # 1 - Air temperature & Relative Humidity from weather station
    # ------------------------------------------------------------------------
    axes[1, 0].plot(data.time, data.ta, color="#009ffd", lw=2.0, label="ta")
    cax1 = divider(axes[1, 0], size="3%", axis="off")
    axes[1, 0].legend(loc="upper left", fontsize=params.lsize)
    axes[1, 0].set_ylabel(r"Temperature [$^{o}C$]", fontsize=params.asize)
    axes[1, 0].set_title("weather station", fontsize=params.lsize, fontstyle="italic")
    axes[1, 0].yaxis.set_major_locator(MultipleLocator(2))
    axes[1, 0].yaxis.set_minor_locator(MultipleLocator(1))
    # hur
    ax12 = axes[1, 0].twinx()
    ax12.plot(data.time, data.hur, color="#ffa400", lw=2.0, label="rh")
    ax12.tick_params(labelsize=params.lsize)
    ax12.yaxis.set_minor_locator(MultipleLocator(1))
    ax12.yaxis.set_major_locator(MultipleLocator(5))
    ax12.legend(loc="upper right", fontsize=params.lsize, borderaxespad=1.0).set_zorder(
        2
    )
    ax12.set_ylabel(r"Relative Humidity [$\%$]", fontsize=params.asize)
    cax12 = divider(ax12, size="3%", axis="off")

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
    cax2 = divider(axes[2, 0], size="3%", axis="off")
    axes[2, 0].legend(loc="lower right", fontsize=params.lsize)
    axes[2, 0].set_ylabel(r"Cumulative rainfall [$mm$]", fontsize=params.asize)
    axes[2, 0].set_title(
        "Disdrometer and Weather station", fontsize=params.lsize, fontstyle="italic"
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
    cbar3.ax.set_ylabel(r"Velocity [$m.s^{-1}$]", fontsize=params.lsize)
    cbar3.ax.tick_params(labelsize=params.lsize)
    axes[0, 1].set_ylim(0, 2500)
    axes[0, 1].set_ylabel("Altitude [m AGL]", fontsize=params.asize)
    axes[0, 1].set_title(
        f"DCR - {data.attrs['radar_source']}", fontsize=params.lsize, fontstyle="italic"
    )
    axes[0, 1].yaxis.set_major_locator(MultipleLocator(500))
    axes[0, 1].yaxis.set_minor_locator(MultipleLocator(100))

    # 4 - wind speed and direction from weather station
    # ------------------------------------------------------------------------
    axes[1, 1].plot(data.time, data.ws, color="r", lw=2.0, label="Wind Speed")
    cax4 = divider(axes[1, 1], size="3%", axis="off")
    axes[1, 1].legend(loc="upper left", fontsize=params.lsize)
    axes[1, 1].set_ylabel(r"Wind Speed [$m.s^{-1}$]", fontsize=params.asize)
    axes[1, 1].set_title("weather station", fontsize=params.lsize, fontstyle="italic")
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
    cax42 = divider(ax42, size="3%", axis="off")
    ax42.tick_params(labelsize=params.lsize)
    ax42.legend(loc="upper right", fontsize=params.lsize)
    ax42.set_ylabel(r"Wind Direction [$^{o}$]", fontsize=params.asize)
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
        axes[2, 1].legend(loc="lower right", fontsize=params.lsize)
        cax5 = divider(axes[2, 1], size="3%", axis="on")
        cbar5 = plt.colorbar(im5[3], cax=cax5, ax=axes[2, 1], ticks=[0, 2, 4, 6, 8, 10])
        cbar5.ax.set_ylabel(r"$\%$ of droplets total", fontsize=params.lsize)
        cbar5.ax.tick_params(labelsize=params.lsize)
    elif y_fit_ok == 0:
        axes[2, 1].text(2, 4, "No data", fontsize=params.asize, fontstyle="italic")
    elif y_fit_ok == -1:
        axes[2, 1].text(2, 4, "No fit", fontsize=params.asize, fontstyle="italic")
        cax5 = divider(axes[2, 1], size="3%", axis="off")
    axes[2, 1].set_xlabel(r"Diameter [$mm$]", fontsize=params.asize)
    axes[2, 1].set_ylabel(r"Fall speed [$m.s^{-1}$]", fontsize=params.asize)
    axes[2, 1].set_xlim(0, 5)
    axes[2, 1].set_ylim(0, 10)
    axes[2, 1].yaxis.set_major_locator(MultipleLocator(2))
    axes[2, 1].yaxis.set_minor_locator(MultipleLocator(1))
    axes[2, 1].xaxis.set_major_locator(MultipleLocator(1))
    axes[2, 1].xaxis.set_minor_locator(MultipleLocator(0.5))

    axes[2, 1].set_title(
        "Relationship disdrometer fall speed / drop size",
        fontsize=params.lsize,
        fontstyle="italic",
    )

    # custom
    # ------------------------------------------------------------------------
    axes[2, 0].set_xlabel("Time [UTC]", fontsize=params.asize)
    axes[1, 1].set_xlabel("Time [UTC]", fontsize=params.asize)
    for n, ax in enumerate(axes.flatten()):
        if ax in [
            axes[2, 0],
            axes[1, 1],
        ]:
            ax.tick_params(labelsize=params.lsize)
            ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(HourLocator(np.arange(0, 24, 3)))
            ax.xaxis.set_minor_locator(HourLocator())
            ax.set_xlim(data.time[0], data.time[-1])
        elif ax == axes[2, 1]:
            ax.tick_params(labelsize=params.lsize)
        else:
            ax.tick_params(labelsize=params.lsize, labelbottom=False)
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
    if data.attrs["location"] in conf.SITES.keys():
        fig.suptitle(
            date.strftime(
                f"Measurement site: {data.attrs['location']} ({data.attrs['geospatial_lat_max']:.3f}N, {data.attrs['geospatial_lon_max']:.3f}E,  {data.attrs['geospatial_vertical_max']:.0f}m)\n{conf.SITES[data.attrs['location']]}\n%d-%m-%Y"
            ),
            fontsize=params.tsize,
        )
    else:
        fig.suptitle(
            date.strftime(
                f"Measurement site: {data.attrs['location']} ({data.attrs['geospatial_lat_max']:.3f}N, {data.attrs['geospatial_lon_max']:.3f}E, {data.attrs['geospatial_vertical_max']:.0f}m)\n%d-%m-%Y"
            ),
            fontsize=params.tsize,
        )
    #
    plt.savefig(output_ql_overview)

    # plt.show()
    plt.close()


def plot_ql_overview_downgraded_mode(
    data: xr.Dataset,
    date: dt.datetime,
    output_ql_overview: str,
    conf: object,
    params: object,
    version: str,
):
    """_summary_

    Args:
        data (xr.Dataset): _description_
        date (dt.datetime): _description_
        output_ql_overview (str): _description_
        conf (object): _description_
        params (object): _description_
        version (str): _description_
    """

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
    cbar0.ax.set_ylabel(r"Zh [$dBZ$]", fontsize=params.lsize)
    cbar0.ax.tick_params(labelsize=params.lsize)
    axes[0, 0].set_ylim(0, 2500)
    axes[0, 0].set_ylabel("Altitude [m AGL]", fontsize=params.asize)
    axes[0, 0].set_title(
        f"DCR - {data.attrs['radar_source']}", fontsize=params.lsize, fontstyle="italic"
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
    cax1 = divider(axes[1, 0], size="3%", axis="off")
    axes[1, 0].legend(loc="lower right", fontsize=params.lsize)
    axes[1, 0].set_ylabel(r"Cumulative rainfall [$mm$]", fontsize=params.asize)
    axes[1, 0].set_title(
        "Disdrometer and Weather station", fontsize=params.lsize, fontstyle="italic"
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
    cbar2.ax.set_ylabel(r"Velocity [$m.s^{-1}$]", fontsize=params.lsize)
    cbar2.ax.tick_params(labelsize=params.lsize)
    axes[0, 1].set_ylim(0, 2500)
    axes[0, 1].set_ylabel("Altitude [m AGL]", fontsize=params.asize)
    axes[0, 1].set_title(
        f"DCR - {data.attrs['radar_source']}", fontsize=params.lsize, fontstyle="italic"
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
        axes[1, 1].legend(loc="lower right", fontsize=params.lsize)
        cax3 = divider(axes[1, 1], size="3%", axis="on")
        cbar3 = plt.colorbar(im3[3], cax=cax3, ax=axes[1, 1], ticks=[0, 2, 4, 6, 8, 10])
        cbar3.ax.set_ylabel(r"$\%$ of droplets total", fontsize=params.lsize)
        cbar3.ax.tick_params(labelsize=params.lsize)
    elif y_fit_ok == 0:
        axes[1, 1].text(2, 4, "No data", fontsize=params.asize, fontstyle="italic")
    elif y_fit_ok == -1:
        axes[1, 1].text(2, 4, "No fit", fontsize=params.asize, fontstyle="italic")
        cax3 = divider(axes[1, 1], size="3%", axis="off")
    axes[1, 1].set_xlabel(r"Diameter [$mm$]", fontsize=params.asize)
    axes[1, 1].set_ylabel(r"Fall speed [$m.s^{-1}$]", fontsize=params.asize)
    axes[1, 1].set_xlim(0, 5)
    axes[1, 1].set_ylim(0, 10)
    axes[1, 1].yaxis.set_major_locator(MultipleLocator(2))
    axes[1, 1].yaxis.set_minor_locator(MultipleLocator(1))
    axes[1, 1].xaxis.set_major_locator(MultipleLocator(1))
    axes[1, 1].xaxis.set_minor_locator(MultipleLocator(0.5))

    axes[1, 1].set_title(
        "Relationship disdrometer fall speed / drop size",
        fontsize=params.lsize,
        fontstyle="italic",
    )

    # custom
    # ------------------------------------------------------------------------
    axes[1, 0].set_xlabel("Time [UTC]", fontsize=params.asize)
    axes[0, 1].set_xlabel("Time [UTC]", fontsize=params.asize)
    for n, ax in enumerate(axes.flatten()):
        if ax != axes[1, 1]:
            ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(HourLocator(np.arange(0, 24, 3)))
            ax.xaxis.set_minor_locator(HourLocator())
            ax.set_xlim(data.time[0], data.time[-1])
        ax.tick_params(labelsize=params.lsize)
        ax.grid(ls="--", alpha=0.5)

    # Final layout & save / display
    # ------------------------------------------------------------------------
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    add_logo()
    fig.align_ylabels()

    if data.attrs["location"] in conf.SITES.keys():
        fig.suptitle(
            date.strftime(
                f"Measurement site: {data.attrs['location']} ({data.attrs['geospatial_lat_max']:.3f}N, {data.attrs['geospatial_lon_max']:.3f}E, {data.attrs['geospatial_vertical_max']:.0f}m)\n{conf.SITES[data.attrs['location']]}\n%d-%m-%Y"
            ),
            fontsize=params.tsize,
        )
    else:
        fig.suptitle(
            date.strftime(
                f"Measurement site: {data.attrs['location']} ({data.attrs['geospatial_lat_max']:.3f}N, {data.attrs['geospatial_lon_max']:.3f}E, {data.attrs['geospatial_vertical_max']:.0f}m)\n%d-%m-%Y"
            ),
            fontsize=params.tsize,
        )

    plt.savefig(output_ql_overview)

    # plt.show()
    plt.close()


def plot_ql_overview_zh(
    data: xr.Dataset,
    date: dt.datetime,
    output_ql_overview_zh: str,
    conf: object,
    params: object,
    version: str,
):
    """_summary_

    Args:
        data (xr.Dataset): _description_
        date (dt.datetime): _description_
        output_dir (str): _description_
        conf (object): _description_
        params (object): _description_
        version (str): _description_
    """
    fig, axes = plt.subplot_mosaic(
        [["top", "top"], ["left", "right"]], constrained_layout=True, figsize=(16, 10)
    )

    # 0 - Zh from DCR and disdrometer
    # ------------------------------------------------------------------------
    ZH_DD = data["Zdlog_vfov_modv_tm"].sel(
        radar_frequencies=data["radar_frequency"].values, method="nearest"
    )
    for r in params.selected_alt:
        ZH_DCR = data["Zdcr"].sel(range=r, method="nearest")
        axes["top"].plot(
            data.time,
            ZH_DCR,
            label=r"$Z_{H}$" + f" from DCR @ {ZH_DCR.range.values:.0f}m",
            lw=2,
        )
    axes["top"].plot(
        data.time, ZH_DD, color="k", label=r"$Z_{H}$ modeled from DD", lw=2
    )
    axes["top"].legend(loc="upper right", fontsize=params.lsize)
    axes["top"].set_ylabel(r"$Z_{H}$ [$dBZ$]", fontsize=params.asize)
    axes["top"].set_title(
        f"Reflectivity from {data.attrs['radar_source']} DCR and {data.attrs['disdrometer_source']} disdrometer",
        fontsize=params.lsize,
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
    params.alt_gate = 150
    ZH_gate = data["Zdcr"].sel(range=params.alt_gate, method="nearest")
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
        axes["left"].legend(loc="upper left", fontsize=params.lsize)
        axes["left"].set_yticks(axes["left"].get_yticks())
        axes["left"].set_yticklabels(np.round(bin_width * axes["left"].get_yticks(), 2))
        axes["left"].set_xlim(xmin, xmax)
    else:
        axes["left"].text(0, 0.5, "No data", fontsize=params.asize, fontstyle="italic")
        axes["left"].set_xlim(0, 1)
    axes["left"].set_xlabel(r"$\Delta$ $Z_{H}$ [$dBZ$]", fontsize=params.asize)
    axes["left"].set_ylabel(r"Rel. Occ [$\%$]", fontsize=params.asize)
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
        + str(params.alt_gate)
        + "m - "
        + r"$Z_{H}$"
        + " from "
        + dd_source,
        fontsize=params.lsize,
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
        ax12.set_ylabel(r"Cum. Occ. [$\%$]", fontsize=params.asize)
        ax12.set_ylim(0, 100)
        ax12.tick_params(labelsize=params.lsize)
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
        f"{conf.INSTRUMENTS['disdrometer'][data.attrs['disdrometer_source']]}"
        + r" DD reflectivity [$dBZ$]",
        fontsize=params.asize,
    )
    axes["right"].set_ylabel(
        f"{conf.INSTRUMENTS['dcr'][data.attrs['radar_source']]}"
        + r" DCR reflectivity [$dBZ$]",
        fontsize=params.asize,
    )
    axes["right"].legend(fontsize=params.lsize)
    axes["right"].yaxis.set_major_locator(MultipleLocator(10))
    axes["right"].yaxis.set_minor_locator(MultipleLocator(5))
    axes["right"].xaxis.set_major_locator(MultipleLocator(10))
    axes["right"].xaxis.set_minor_locator(MultipleLocator(5))

    # custom
    # ------------------------------------------------------------------------
    for pos in list(axes):
        axes[pos].grid(ls="--", alpha=0.5)
        axes[pos].tick_params(labelsize=params.lsize)

    # Final layout & save / display
    # ------------------------------------------------------------------------
    plt.tight_layout()
    plt.subplots_adjust(top=0.86, hspace=0.2, wspace=0.15)
    add_logo()
    fig.align_ylabels()

    if data.attrs["location"] in conf.SITES.keys():
        fig.suptitle(
            date.strftime(
                f"Measurement site: {data.attrs['location']} ({data.attrs['geospatial_lat_max']:.3f}N, {data.attrs['geospatial_lon_max']:.3f}E, {data.attrs['geospatial_vertical_max']:.0f}m)\n{conf.SITES[data.attrs['location']]}\n%d-%m-%Y"
            ),
            fontsize=params.tsize,
        )
    else:
        fig.suptitle(
            date.strftime(
                f"Measurement site: {data.attrs['location']} ({data.attrs['geospatial_lat_max']:.3f}N, {data.attrs['geospatial_lon_max']:.3f}E, {data.attrs['geospatial_vertical_max']:.0f}m)\n%d-%m-%Y"
            ),
            fontsize=params.tsize,
        )

    plt.savefig(output_ql_overview_zh)

    # plt.show()
    plt.close()
