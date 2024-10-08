# Input : xarray.Dataset of the preprocessed Ze data over a certain period ;
# Output : quicklooks for the given period


import datetime as dt
import logging
import os

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from create_input_files_quicklooks import get_data_event
from matplotlib.gridspec import GridSpec
from rain_event_selection import selection
from scipy.optimize import curve_fit
from scipy.stats import cumfreq
from sklearn.linear_model import LinearRegression

lgr = logging.getLogger(__name__)


pd.options.mode.chained_assignment = None  # default='warn'

CHUNK_THICKNESS = 15  # minutes

MN = 60  # mn : Time horizon before and after therain event
MAX_WS = 7  # m/s : Wind speed threshold for QC/QF
MIN_T = 2  # °C : Min temperature threshold for QC/QF
MIN_CUM = 3  # mm : minimum rain accumulation to keep an event in the statistics
MAX_RR = 3

CUM_REL_ERROR = 0.3
# 1, max relative error in rain acc. between disdro and pluvio for an event
FALLSPEED_REL_ERROR = 0.3
# 1, relative difference between theoretical rain fall speed and disdro fall speed

TIMESTAMP_THRESHOLDS = [MIN_T, MAX_WS, MAX_RR, FALLSPEED_REL_ERROR]
EVENT_THRESHOLDS = [MIN_CUM, CUM_REL_ERROR]

DELTA_DISDRO = dt.timedelta(
    minutes=MN
)  # check the rain behaviour before and after event
# Before event : bias before the first trigger of the pluviometer


def compute_quality_matrix():
    pass


def quicklooks(
    weather,
    dcr,
    disdro,
    output_path,
    thresholds=TIMESTAMP_THRESHOLDS + EVENT_THRESHOLDS,
):  # noqa: C901
    # "disdro" : preprocessed data ;
    # the three files are outputs from methods data_xxx_event
    # None if the file cannot be built by these methods

    # Create a "time" that we call when necessary in the plots ?
    # (First check that the three time vectors match perfectly ?)
    if (weather is None) or (disdro is None) or (dcr is None):
        return None

    try:
        # Disdro and DCR resampled reflectivity data at the good range
        Z_dcr = dcr.Zh.isel({"range": [3, 4, 6, 8]})
        z_disdro = disdro.Ze_tm
        z_disdro[np.where(z_disdro == 0)] = np.nan  # avoid np.inf in Z_disdro
        Z_disdro = 10 * np.log10(z_disdro)

        start_time = pd.Timestamp(Z_disdro.time.values[0]).round(freq="1T")
        end_time = pd.Timestamp(Z_disdro.time.values[-1]).round(freq="1T")

        time_index = pd.date_range(
            start_time, end_time + pd.Timedelta(minutes=1), freq="1T"
        )
        time_index_offset = time_index - pd.Timedelta(30, "sec")

        Z_dcr_resampled = Z_dcr.groupby_bins(
            "time", time_index_offset, labels=time_index[:-1]
        ).median(dim="time", keep_attrs=True)
        Z_dcr_resampled = Z_dcr_resampled.rename({"time_bins": "time"})
        Z_dcr_200m_resampled = Z_dcr_resampled[:, 3]  # data @ 212.5m, sampling 1 minute

        Doppler_resampled = (
            dcr.v.isel({"range": [4, 6, 8]})
            .groupby_bins("time", time_index_offset, labels=time_index[:-1])
            .mean(dim="time", keep_attrs=True)
        )
        Doppler_resampled = Doppler_resampled.rename({"time_bins": "time"})

        # Date format for plots
        locator = mpl.dates.AutoDateLocator()
        formatter = mpl.dates.ConciseDateFormatter(locator)

        # Plot 1 :
        fig, axes = plt.subplots(5, 1, figsize=(12, 20))
        (ax1, ax2, ax3, ax4, ax5) = axes
        plt.subplots_adjust(0.08, 0.04, 0.89, 0.94, hspace=0.27)

        for axe in axes.flatten():
            axe.xaxis.set_major_formatter(formatter)
            axe.set_xlabel("Time (UTC)")

        cmap = plt.get_cmap("rainbow").copy()
        cmap.set_under("w")

        # 1.1. plot 2D reflectivity
        Z_2500m = dcr.Zh.isel({"range": np.arange(0, 100)})
        pc = ax1.pcolormesh(
            pd.to_datetime(Z_2500m.time.values),
            Z_2500m.range.values,
            Z_2500m.T,
            vmin=0,
            cmap=cmap,
            shading="nearest",
        )
        ax1.set_ylabel(dcr.Zh.range.attrs["long_name"])
        ax1.set_title("DCR 94 GHz reflectivity")  # "94 GHz" should vary
        pos = ax1.get_position()
        cb_ax = fig.add_axes([0.91, pos.y0, 0.02, pos.y1 - pos.y0])
        cb1 = fig.colorbar(pc, orientation="vertical", cax=cb_ax)
        cb1.set_label(dcr.Zh.attrs["long_name"] + " (dBZ)")
        # 1.2. plot 2D doppler velocity
        Doppler_2500m = dcr.v.isel({"range": np.arange(0, 100)})
        pc2 = ax2.pcolormesh(
            pd.to_datetime(Doppler_2500m.time.values),
            Doppler_2500m.range.values,
            Doppler_2500m.T,
            vmax=0,
            vmin=-5,
            cmap=cmap,
            shading="nearest",
        )
        ax2.set_ylabel(dcr.v.range.attrs["long_name"])
        ax2.set_title("DCR Doppler velocity")
        pos = ax2.get_position()
        cb_ax2 = fig.add_axes([0.91, pos.y0, 0.02, pos.y1 - pos.y0])
        cb2 = fig.colorbar(pc2, orientation="vertical", cax=cb_ax2)
        cb2.set_label(dcr.v.attrs["long_name"] + " (m/s)")
        # 1.3. plot rain accumulation : weather station VS disdrometer
        ax3.plot(
            weather.time, weather["rain_sum"], color="red", label="Weather station"
        )
        ax3.plot(
            disdro.time[:],
            disdro["disdro_rain_sum"][:],
            color="green",
            label="Disdrometer",
        )  # to be checked : good values in 'cp' or not ?
        ax3.legend()
        ax3.set_ylabel("Precipitation [mm]")
        ax3.set_title("Rain accumulation")
        ax3.grid()
        ax3.set_xlim(
            left=Z_dcr_resampled.time.values.min(),
            right=Z_dcr_resampled.time.values.max(),
        )

        # 1.4. Air temperature timeseries
        ax4.plot(weather.time, weather["air_temperature"])
        ax4.set_title("Air temperature")
        ax4.set_ylabel("T [K]")
        ax4.grid()
        ax4.set_xlim(
            left=Z_dcr_resampled.time.values.min(),
            right=Z_dcr_resampled.time.values.max(),
        )

        ax5.plot(weather.time, weather["wind_speed"])
        ax5.set_title("Wind Speed")
        ax5.set_ylabel("wind speed [m/s]")
        ax5.grid()
        ax5.set_xlim(
            left=Z_dcr_resampled.time.values.min(),
            right=Z_dcr_resampled.time.values.max(),
        )

        fig.text(
            s=start_time.strftime("%Y-%m-%d") + " : Overview of the event",
            fontsize=18,
            horizontalalignment="center",
            verticalalignment="center",
            y=0.97,
            x=0.5,
        )

        if not os.path.exists(
            output_path + "/{}".format(start_time.strftime("%Y%m%d"))
        ):
            os.makedirs(output_path + "/{}".format(start_time.strftime("%Y%m%d")))

        plt.savefig(
            output_path
            + "/{}/{}_quicklook1.png".format(
                start_time.strftime("%Y%m%d"), start_time.strftime("%Y%m%d")
            ),
            dpi=500,
            transparent=False,
            facecolor="white",
        )
        # plt.show(block=False)

        # Plot 2 :
        fig = plt.figure(figsize=(12, 20))
        plt.subplots_adjust(0.08, 0.04, 0.89, 0.94, hspace=0.27)

        gs = GridSpec(4, 3, figure=fig)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :])
        ax3 = fig.add_subplot(gs[2, :])
        ax4 = fig.add_subplot(gs[3, 1])

        for axe in (ax1, ax2):
            axe.xaxis.set_major_formatter(formatter)
            axe.set_xlabel("Time (UTC)")

        # 2.1. Compare Z from disdro and DCR @ different altitudes
        ax1.plot(disdro.time[:], Z_disdro[:], label="Disdrometer", color="green")
        for i in range(Z_dcr.shape[1]):
            ax1.plot(
                Z_dcr_resampled.time[:],
                Z_dcr_resampled[:, i],
                label=f"DCR @ {Z_dcr_resampled.range[i].values}m",
                linewidth=0.5,
            )
        ax1.grid()
        ax1.legend()
        ax1.set_ylabel("Z [dBZ]")
        ax1.set_title("Reflectivity from DCR and disdrometers")
        ax1.set_xlim(
            left=Z_dcr_resampled.time.values.min(),
            right=Z_dcr_resampled.time.values.max(),
        )
        # 2.2. Reflectivity differences
        ax2.plot(
            Z_dcr_200m_resampled.time[:],
            Z_dcr_200m_resampled[:] - Z_disdro[:],
            label="$Z_{DCR}$ - $Z_{Disdrometer}$",
            color="green",
        )
        ax2.grid()
        ax2.legend()
        ax2.set_ylabel(r"$ \Delta Z [dBZ]$")
        ax2.set_title(
            f"Differences of reflectivity : DCR @ {Z_dcr_200m_resampled.range.values}m vs Disdrometer"  # noqa: E501
        )
        ax2.set_xlim(
            left=Z_dcr_resampled.time.values.min(),
            right=Z_dcr_resampled.time.values.max(),
        )
        # 2.3. PDF / empirical CDF for Delta Z :
        f = np.where((np.isfinite(Z_dcr_200m_resampled)) & (np.isfinite(Z_disdro)))[0]
        cdf = cumfreq(Z_dcr_200m_resampled[f] - Z_disdro[f], numbins=100)
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
        # print(np.nanmedian(Z_dcr_200m_resampled[:] - Z_disdro[:]),
        # np.where( (np.isfinite(Z_dcr_200m_resampled) )
        # & ( np.isfinite(Z_disdro) ) )[0].shape )
        ax3.hist(
            Z_dcr_200m_resampled[:] - Z_disdro[:],
            label=f"$Z_{{DCR}}$ - $Z_{{Disdrometer}}$, $Med = ${np.nanmedian(Z_dcr_200m_resampled[:] - Z_disdro[:]):.2f} $dBZ$",  # noqa: E501
            alpha=0.4,
            color="green",
            density=True,
            bins=int(
                (
                    (Z_dcr_200m_resampled[:] - Z_disdro[:]).max()
                    - (Z_dcr_200m_resampled[:] - Z_disdro[:]).min()
                )
                / 0.1
            ),
        )
        ax3.axvline(x=(Z_dcr_200m_resampled[:] - Z_disdro[:]).median(), color="green")
        ax3.legend(loc="upper left")
        ax3.grid()
        ax3.set_xlim(left=-15, right=15)
        ax3.set_xlabel(r"$\Delta Z [dBZ]$")
        ax3.set_ylabel("Density")
        ax3_cdf.set_ylabel("% of values")
        ax3.set_title(
            "{} -- PDF of differences DCR / Disdro".format(
                start_time.strftime("%Y%m%d")
            )
        )
        # 2.4. Compare Z(DCR) VS Z(disdro)
        model = LinearRegression()
        x_t = Z_disdro[MN : -MN - 1].values
        y_t = Z_dcr_200m_resampled[MN : -MN - 1].values
        filt = np.where((np.isfinite(x_t)) & (np.isfinite(y_t)))[0]
        x_t = x_t[filt].reshape((-1, 1))
        y_t = y_t[filt].reshape((-1, 1))
        # print(x_t.shape)
        model.fit(x_t, y_t)
        y_hat = model.predict(x_t)
        ax4.plot(
            x_t,
            y_hat,
            color="C1",
            label=f"$Z_{{DCR}}$ = {model.coef_[0][0]:.3f} $Z_{{DD}}$ + {model.intercept_[0]:.3f}",  # noqa: E501
        )
        ax4.text(
            1,
            18,
            f"$R^2$ = {model.score(x_t, y_t):.2f}",
            fontweight="semibold",
            fontsize=10,
        )
        ax4.scatter(x_t, y_t, color="green")

        ax4.plot([0, 25], [0, 25], linestyle="--", color="grey", label="y = x")
        ax4.set_xlim(left=-0, right=25)
        ax4.set_ylim(bottom=-0, top=25)
        ax4.grid()
        ax4.legend(loc="upper left")
        ax4.set_xlabel("$Z_{{DD}}$ (dBZ)")
        ax4.set_ylabel("$Z_{{DCR}}$ (dBZ)")
        ax4.set_title("Scatterplot $Z_{{DD}}$ vs $Z_{{DCR}}$")
        ax4.set_aspect("equal")

        fig.text(
            s=start_time.strftime("%Y-%m-%d") + ": Reflectivity data",
            fontsize=18,
            horizontalalignment="center",
            verticalalignment="center",
            y=0.97,
            x=0.5,
        )

        plt.savefig(
            output_path
            + "/{}/{}_quicklook2.png".format(
                start_time.strftime("%Y%m%d"),
                start_time.strftime("%Y%m%d"),
            ),
            dpi=500,
            transparent=False,
            facecolor="white",
        )
        # plt.show(block=False)

        # Plot 3 : Doppler velocities
        fig = plt.figure(figsize=(12, 20))
        plt.subplots_adjust(0.08, 0.04, 0.89, 0.94, hspace=0.27)

        gs = GridSpec(4, 3, figure=fig)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :])
        ax3 = fig.add_subplot(gs[2, :])
        ax4 = fig.add_subplot(gs[3, 1])

        for axe in (ax1, ax2):
            axe.xaxis.set_major_formatter(formatter)
            axe.set_xlabel("Time (UTC)")

        # 3.1. Disdrometer fall speed vs DCR doppler velocity
        disdro_tr = np.transpose(
            disdro["psd"].values, axes=(0, 2, 1)
        )  # Because for the moment psd size and speed axes are exchanged
        disdro_fallspeed = np.zeros(disdro_tr.shape[0])
        for t in range(len(disdro_fallspeed)):
            drops_per_time_and_speed = np.nansum(disdro_tr[t, :, :], axis=0).flatten()
            disdro_fallspeed[t] = np.nansum(
                disdro["speed_classes"] * drops_per_time_and_speed
            ) / np.nansum(disdro_tr[t, :, :])
        print("disdro fallspeed vector shape :", disdro_fallspeed.shape)
        print(disdro.time.values[0], Doppler_resampled.time.values[0], start_time)
        print(disdro.time.values.shape, Doppler_resampled.time.values.shape)
        # Time vectors : match OK !

        # There still remains to compute standard deviation to be able to plot it
        for i in range(3):
            filter_fallspeed = np.where(
                (np.isfinite(disdro_fallspeed)) & (np.isfinite(Doppler_resampled[:, i]))
            )[0]
            print("filter isfinite : shape = ", filter_fallspeed.shape)
            pearsonr = stats.pearsonr(
                disdro_fallspeed[filter_fallspeed],
                Doppler_resampled[:, i][filter_fallspeed],
            )
            ax1.plot(
                weather.time,
                -Doppler_resampled[:, i],
                label=f"DCR @ {Doppler_resampled.range[i].values}m, corr = {pearsonr[0]:.2f}",  # noqa: E501
                linewidth=0.5,
            )
        ax1.plot(
            disdro.time,
            disdro_fallspeed,
            label="disdrometer avg fall speed",
            color="green",
        )
        ax1.grid()
        ax1.set_ylabel("Fall speed [m/s]")
        ax1.legend()
        ax1.set_title("Disdrometer and DCR fall speed timeseries")
        ax1.set_xlim(
            left=Z_dcr_resampled.time.values.min(),
            right=Z_dcr_resampled.time.values.max(),
        )
        # 3.2. Delta fall speed
        ax2.plot(
            weather.time,
            -Doppler_resampled[:, 2] - disdro_fallspeed,
            label=f"DCR @ {Doppler_resampled.range[2].values}m - Disdrometer",
            color="green",
        )
        ax2.grid()
        ax2.legend()
        ax2.set_ylabel(r"$\Delta V$ (m/s)")
        ax2.set_title(" DCR / Disdrometer fall speed differences")
        ax2.set_xlim(
            left=Z_dcr_resampled.time.values.min(),
            right=Z_dcr_resampled.time.values.max(),
        )
        # 3.3. PDF Delta V
        f = np.where(
            (np.isfinite(Doppler_resampled[:, 2])) & (np.isfinite(disdro_fallspeed))
        )[0]
        cdf = cumfreq(-Doppler_resampled[:, 2][f] - disdro_fallspeed[f], numbins=100)
        dv = cdf.lowerlimit + np.linspace(
            0, cdf.binsize * cdf.cumcount.size, cdf.cumcount.size
        )
        ax3_cdf = ax3.twinx()
        ax3_cdf.set_ylim(top=1, bottom=0)
        ax3_cdf.plot(
            dv,
            cdf.cumcount / len(f),
            label=r"$CDF \Delta V_{DCR/Disdrometer}$",
            marker="o",
            color="green",
        )
        ax3_cdf.set_ylabel("Empirical CDF")
        ax3.hist(
            -Doppler_resampled[:, 2] - disdro_fallspeed,
            density=True,
            label=f"DCR - Disdrometer, $Med = ${np.nanmedian(-Doppler_resampled[:, 2] - disdro_fallspeed):.2f} $m/s$",  # noqa: E501
            alpha=0.4,
            color="green",
            bins=int(
                (
                    (-Doppler_resampled[:, 2] - disdro_fallspeed).max()
                    - (-Doppler_resampled[:, 2] - disdro_fallspeed).min()
                )
                / 0.1
            ),
        )
        ax3.axvline(
            x=(-Doppler_resampled[:, 2] - disdro_fallspeed).median(), color="green"
        )
        ax3.set_xlim(left=-4, right=4)
        ax3.legend(loc="upper left")
        ax3.set_xlabel(r"$\Delta V$ [m/s]")
        ax3.set_ylabel("Density")
        ax3.set_title("PDF of fall speed differences")
        ax3.grid()
        # 3.4. Scatterplot Doppler velocity vs disdro fallspeed + Regression
        model = LinearRegression()
        x_t = disdro_fallspeed[MN:-MN]
        y_t = -Doppler_resampled[MN:-MN, 1].values
        filt_fallspeed = np.where((np.isfinite(x_t)) & (np.isfinite(y_t)))[0]
        print(filt_fallspeed.shape)
        x_t = x_t[filt_fallspeed].reshape((-1, 1))
        y_t = y_t[filt_fallspeed].reshape((-1, 1))

        model.fit(x_t, y_t)
        y_hat = model.predict(x_t)

        ax4.scatter(x_t, y_t, color="green")
        lab = f"$V_{{DCR}}$ = {model.coef_[0][0]:.3f} $V_{{DD}}$ + {model.intercept_[0]:.3f}"  # noqa: E501
        ax4.plot(
            x_t,
            y_hat,
            color="C1",
            label=lab,
        )
        ax4.text(
            0.2,
            1.3,
            f"$R^2$={model.score(x_t, y_t):.2f}",
            fontsize=10,
            fontweight="semibold",
        )
        ax4.plot([0, 25], [0, 25], linestyle="--", color="grey", label="y = x")
        ax4.set_xlim(left=-0, right=6)
        ax4.set_ylim(bottom=-0, top=6)
        ax4.grid()
        ax4.set_aspect("equal")
        ax4.set_xlabel("$V_{Disdrometer}$ (m/s)")
        ax4.set_ylabel("$V_{DCR}$ (m/s)")
        ax4.legend(loc="lower left")
        ax4.set_title("Scatterplot $V_{{DCR}}$ vs $V_{{Disdrometer}}$")

        fig.text(
            s=start_time.strftime("%Y-%m-%d") + ": Rain fall speed data",
            fontsize=18,
            horizontalalignment="center",
            verticalalignment="center",
            y=0.97,
            x=0.5,
        )
        plt.savefig(
            output_path
            + "/{}/{}_quicklook3.png".format(
                start_time.strftime("%Y%m%d"), start_time.strftime("%Y%m%d")
            ),
            dpi=500,
            transparent=False,
            facecolor="white",
        )
        # plt.show(block = False)

        # Plot 4 :
        fig = plt.figure(figsize=(12, 20))
        plt.subplots_adjust(0.08, 0.04, 0.89, 0.94, hspace=0.27)

        gs = GridSpec(5, 4, figure=fig)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :])
        ax3 = fig.add_subplot(gs[2, 1:3])
        ax4 = fig.add_subplot(gs[3, :])
        ax5 = fig.add_subplot(gs[4, :])

        for axe in (ax1, ax2, ax4, ax5):
            axe.xaxis.set_major_formatter(formatter)
            axe.set_xlabel("Time (UTC)")

        # 4.1. Rain accumulation differences
        Delta_cum_pluvio_disdro = (
            disdro["disdro_rain_sum"][:] - weather["rain_sum"]
        ) / weather["rain_sum"]
        ax1.plot(
            weather.time,
            Delta_cum_pluvio_disdro * 100,
            color="green",
            label="Disdrometer - Weather station",
        )
        ax1.legend()
        ax1.set_ylabel("Difference of rain accumulation (%)")
        ax1.grid()
        ax1.set_xlim(
            left=Z_dcr_resampled.time.values.min(),
            right=Z_dcr_resampled.time.values.max(),
        )
        ax1.set_title("Rain accumulation ratio Disdrometer v. Pluviometer")

        # # 4.2. Fall speed DCR/Disdro : relative difference
        # Fall_speed_rel_error = (
        #     -Doppler_resampled[:, 2] - disdro_fallspeed
        # ) / disdro_fallspeed
        # ax2.plot(Doppler_resampled.time, Fall_speed_rel_error * 100)
        # ax2.grid()
        # ax2.set_ylabel("Relative error (%)")
        # ax2.set_xlim(
        #     left=Z_dcr_resampled.time.values.min(),
        #     right=Z_dcr_resampled.time.values.max(),
        # )
        # ax2.set_title("Fall speed relative error DCR v. Disdrometer")

        # 4.2. Wind direction time series
        ax2.set_ylim(0, 360)
        ax2.plot(weather.time, weather.wind_direction)
        ax2.grid()
        if (weather.main_wind_dir % 180 > 45) and (weather.main_wind_dir % 180 < 135):
            ax2.axhspan(
                ymin=weather.main_wind_dir - 45,
                ymax=weather.main_wind_dir + 45,
                color="green",
                alpha=0.3,
            )
            ax2.axhspan(
                ymin=(weather.main_wind_dir + 180) % 360 - 45,
                ymax=(weather.main_wind_dir + 180) % 360 + 45,
                color="green",
                alpha=0.3,
            )
        else:
            mwd1, mwd2 = weather.main_wind_dir, (weather.main_wind_dir + 180) % 360
            if np.abs(mwd1 - 360) > 45:
                mwd1, mwd2 = mwd2, mwd1
            ax2.axhspan(
                ymin=(mwd2) - 45,
                ymax=(mwd2) + 45,
                color="green",
                alpha=0.3,
            )
            edges = [(mwd1 - 45) % 360, (mwd1 + 45) % 360]
            ax2.axhspan(
                ymin=edges[0],
                ymax=360,
                color="green",
                alpha=0.3,
            )
            ax2.axhspan(
                ymin=0,
                ymax=edges[1],
                color="green",
                alpha=0.3,
            )

        ax2.set_ylabel("Wind direction [°]")
        ax2.set_xlim(left=weather.time.values.min(), right=weather.time.values.max())
        ax2.set_title("Wind direction over the event")

        # 4.3. Plot relationship Fallspeed(Diameter) and compare it to theoretical value
        def f_th(x):
            return 9.40 * (
                1 - np.exp(-1.57 * (10**3) * np.power(x * (10**-3), 1.15))
            )  # Gun and Kinzer (th.)

        def f_fit(x, a, b, c):
            return a * (1 - np.exp(-b * np.power(x * (10**-3), c)))  # target shape

        drop_density = np.nansum(
            disdro_tr, axis=0
        )  # sum over time dim  # transpose if data is switched
        psd_nonzero = np.where(drop_density != 0)
        x, y = [], []

        print(np.where(np.isnan(disdro_tr)))  ## à minuit !
        for k in range(
            len(psd_nonzero[0])
        ):  # add observations (size, speed) in the proportions described by the psd
            x += [disdro["size_classes"][psd_nonzero[0][k]]] * int(
                drop_density[psd_nonzero[0][k], psd_nonzero[1][k]]
            )
            y += [disdro["speed_classes"][psd_nonzero[1][k]]] * int(
                drop_density[psd_nonzero[0][k], psd_nonzero[1][k]]
            )

        X, Y = np.array(x), np.array(y)

        popt, pcov = curve_fit(f_fit, X, Y)
        y_hat = f_fit(disdro["size_classes"], popt[0], popt[1], popt[2])
        y_th = f_th(disdro["size_classes"])
        y_th_disdro = y_th.copy()

        h2 = ax3.hist2d(
            X,
            Y,
            cmin=len(X) / 1000,
            bins=[disdro["size_classes"], disdro["speed_classes"]],
            density=False,
        )
        ax3.plot(
            disdro["size_classes"],
            y_hat,
            c="green",
            label="Fit on DD measurements",
        )
        ax3.plot(
            disdro["size_classes"],
            y_th,
            c="C1",
            label="Fall speed model (Gun and Kinzer)",
        )
        fig.colorbar(h2[3], ax=ax3)
        ax3.legend(loc="best")
        ax3.grid()
        ax3.set_xlabel("Diameter (mm)")
        ax3.set_ylabel("Fall speed (m/s)")
        # ax3.set_xlim(disdro["size_classes"].min(), disdro["size_classes"].max())
        # ax3.set_ylim(disdro["speed_classes"].min(), disdro["speed_classes"].max())
        ax3.set_xlim(0, 5)
        ax3.set_ylim(0, 10)
        ax3.set_title("Relationship disdrometer fall speed / drop size ")
        # 4.4. Relative difference : disdro fallspeed vs speed law
        ratio_vdisdro_vth = np.zeros(len(disdro.time))
        for t in range(len(disdro.time)):
            drops_per_time_and_diameter = np.nansum(
                disdro_tr[t, :, :], axis=1
            ).flatten()
            mu_d = drops_per_time_and_diameter / np.nansum(drops_per_time_and_diameter)

            vth_disdro = np.nansum(mu_d * y_th_disdro)
            vobs_disdro = disdro_fallspeed[t]
            ratio_vdisdro_vth[t] = vobs_disdro / vth_disdro

        ax4.plot(
            disdro.time, ratio_vdisdro_vth * 100, label="Disdrometer", color="green"
        )
        ax4.grid()
        ax4.legend()
        ax4.set_ylabel("Ratio (%)")
        ax4.set_xlim(
            left=Z_dcr_resampled.time.values.min(),
            right=Z_dcr_resampled.time.values.max(),
        )
        ax4.set_title(
            "Ratio - AVG Fall Speed measured by disdrometers v. Gun and Kinzer law"
        )
        ax4.set_ylim(top=150, bottom=50)
        # 4.5. Quality checks / Quality flags : time series
        QC_delta_cum = (np.abs(Delta_cum_pluvio_disdro) <= 0.3).values.reshape((-1, 1))
        QF_meteo = np.vstack(
            (
                weather.QF_T.values,
                weather.QF_ws.values,
                weather.QF_wd.values,
                weather.QF_acc.values,
                weather.QF_RR.values,
            )
        ).T
        QF_meteo = QF_meteo[:, [0, 1, 2, 4, 3]]
        QC_vdsd_t = (np.abs(ratio_vdisdro_vth - 1) <= 0.3).reshape((-1, 1))
        # Quid QC sur fallspeed ? Wind direction (régime de bon fonctionnement du DD) ?
        Quality_matrix = np.hstack((QF_meteo, QC_delta_cum, QC_vdsd_t))
        Quality_sum = Quality_matrix[:, [0, 1, 2, 3, 6]].all(axis=1)

        Quality_matrix_sum = np.flip(
            np.hstack((Quality_matrix, Quality_sum.reshape((-1, 1)))), axis=1
        )
        Quality_matrix_sum.astype(int)
        Quality_matrix_sum = Quality_matrix_sum[:, [0, 1, 4, 2, 3, 5, 6, 7]]

        t = weather.time
        cmap = colors.ListedColormap(["red", "green"])
        cmap2 = colors.ListedColormap(["grey", "green"])

        ax5.pcolormesh(
            t,
            np.arange(len(Quality_matrix_sum.T)),
            Quality_matrix_sum.T,
            cmap=cmap,
            shading="nearest",
        )
        ax5.pcolormesh(
            t,
            np.arange(3, 6),
            Quality_matrix_sum[:, 3:6].T,
            cmap=cmap2,
            shading="nearest",
        )

        ax5.xaxis.set_major_locator(mpl.dates.HourLocator(interval=1))
        ax5.yaxis.set_label_position("right")
        ax5.yaxis.tick_right()
        ax5.set_yticks([q for q in range(len(Quality_matrix_sum.T))])
        ax5.set_yticklabels(
            np.array(
                [
                    "QC T°",
                    "QC wind sp",
                    "QC wind dir",
                    "QC rain acc.",
                    "QF Pluvio/Disdro",
                    "QC Rain Rate",
                    "QC V(D)",
                    r"$\Pi$",
                ]
            )[::-1],
            fontsize=8,
        )
        ax5.set_title("QF / QC timeseries")

        fig.text(
            s=start_time.strftime("%Y-%m-%d") + ": Quality checks",
            fontsize=18,
            horizontalalignment="center",
            verticalalignment="center",
            y=0.97,
            x=0.5,
        )
        print("### QL4 OK ! ###")

        plt.savefig(
            output_path
            + "/{}/{}_quicklook4.png".format(
                start_time.strftime("%Y%m%d"), start_time.strftime("%Y%m%d")
            ),
            dpi=500,
            transparent=False,
            facecolor="white",
        )

        # plt.show(block = False)

    except RuntimeError:
        return None

    return True


if __name__ == "__main__":
    events = selection()
    s = events["Start_time"]
    e = events["End_time"]
    w, r, dd = get_data_event(s[0], e[0])
    output = "/home/ygrit/Documents/disdro_processing/ccres_disdrometer_processing/quicklooks"  # noqa: E501
    quicklooks(w, r, dd, output)
