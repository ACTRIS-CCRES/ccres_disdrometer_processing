import datetime as dt
import glob
import logging

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from create_input_files_quicklooks import data_dcr_event

lgr = logging.getLogger(__name__)

CLIC = 0.001  # mm/mn : can identify rain rates lower than 0.1mm/h for disdro data
RRMAX = 3  # mm/h
MIN_CUM = 2  # mm/episode
MAX_MEANWS = 7
MAX_WS = 10
MIN_T = 2  # °C

DELTA_EVENT = 60
DELTA_LENGTH = 180
CHUNK_THICKNESS = 15  # mn

MN = 60
DELTA_DISDRO = dt.timedelta(minutes=MN)


def dz_per_event(
    preprocessed_ds,
    data_dir,
    start_time,
    end_time,
    filtered=True,
):
    # maybe should write a method "data_preprocessed_event"
    # with the same structure as data extractions for the different instruments
    # used before
    data = preprocessed_ds.sel(
        {"time": slice(start_time - DELTA_DISDRO, end_time + DELTA_DISDRO)}
    )
    if data is None:  # if len(data) != timedelta(end-start)+2 Delta ?
        return None

    try:
        # Get data
        Z_dcr = data.Zh.isel(
            {"range": [0, 2, 4, 5]}
        )  # 36m, 108m, 180m, 216m (108m altitude to remove)
        z_disdro = data.Ze_tm
        z_disdro[np.where(z_disdro == 0)] = np.nan  # avoid np.inf in Z_disdro
        Z_disdro = 10 * np.log10(z_disdro)
        time_index = pd.date_range(
            start_time - DELTA_DISDRO,
            end_time + DELTA_DISDRO + pd.Timedelta(minutes=1),
            freq="1T",
        )
        time_index_offset = time_index - pd.Timedelta(30, "sec")

        Z_dcr_resampled = Z_dcr.groupby_bins(
            "time", time_index_offset, labels=time_index[:-1]
        ).median(dim="time", keep_attrs=True)
        Z_dcr_resampled = Z_dcr_resampled.rename({"time_bins": "time"})
        Z_dcr_200m_resampled = Z_dcr_resampled[:, 2]  # data @ ~215m, sampling 1 minute

        print(
            "Len of time vector (delta_disdro included)",
            len(time_index) - 1,
            Z_dcr_200m_resampled.shape,
            Z_disdro.shape,
        )
        if len(Z_dcr_200m_resampled) != len(Z_disdro):
            Z_dcr_200m_resampled = Z_dcr_200m_resampled.sel(
                {"time": Z_disdro.time.values}
            )
            Z_dcr_resampled = Z_dcr_resampled.sel({"time": Z_disdro.time.values})
            print("TIME VECTOR MODIFIED")
        # Doppler = data.v.isel({"range": [1, 4, 6, 8]})
        # Doppler_resampled = Z_dcr.groupby_bins(
        #     "time", time_index_offset, labels=time_index[:-1]
        # ).mean(dim="time", keep_attrs=True)

        if (
            len(
                np.where((np.isfinite(Z_dcr_200m_resampled)) & (np.isfinite(Z_disdro)))[
                    0
                ]
            )
            == 0
        ):
            lgr.CRITICAL(
                "Problem : no finite reflectivity data for dcr/disdro comparison"
            )
            return None

        # QL Plot Zbasta vs Zdisdro
        fig, ax = plt.subplots()
        locator = mpl.dates.AutoDateLocator()
        formatter = mpl.dates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_formatter(formatter)
        for i in range(len(Z_dcr_resampled.range.values)):
            rng = Z_dcr_resampled.range.values[i] - dcr.altitude.values[0]
            # print(
            #     type(rng),
            #     type(Z_dcr_resampled.range.values[i]),
            #     type(dcr.altitude.values[0]),
            #     dcr.altitude.values.shape,
            # )
            ax.plot(
                data.time.values,
                Z_dcr_resampled[:, i].values,
                label="radar @ {:.0f} m".format(rng),
                linewidth=1,
            )
        ax.plot(
            data.time.values,
            Z_disdro,
            color="green",
            label="disdrometer reflectivity",
        )
        plt.ylabel("Z [dBZ]")
        plt.xlabel("Time (UTC)")
        plt.grid()
        plt.legend()
        plt.savefig(
            data_dir + "/QL/{}_Refl.png".format(start_time.strftime("%Y%m%d%T")),
            dpi=500,
            transparent=False,
            edgecolor="white",
        )
        plt.close()

        # QL Plot Disdro rain
        fig, ax = plt.subplots()
        locator = mpl.dates.AutoDateLocator()
        formatter = mpl.dates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.plot(
            data.time.values,
            np.cumsum(data.pr / 60),
            color="green",
        )
        plt.ylabel("Rain accumulation [mm]")
        plt.xlabel("Time (UTC)")
        plt.grid()
        plt.savefig(
            data_dir + "/QL/{}_cumsum.png".format(start_time.strftime("%Y%m%d%T")),
            dpi=500,
            transparent=False,
            edgecolor="white",
        )
        plt.close()

        # Delta Z basta / disdro

        dZdisdro = Z_dcr_200m_resampled[:] - Z_disdro[:]
        print("shape of the DZ vector", dZdisdro.shape)

        disdro_tr = np.transpose(
            data["psd"].values, axes=(0, 1, 2)
        )  # bNo more need to transpose now it is done in the preprocessing !
        disdro_fallspeed = np.zeros(disdro_tr.shape[0])
        for t in range(len(disdro_fallspeed)):
            drops_per_time_and_speed = np.nansum(disdro_tr[t, :, :], axis=0).flatten()
            disdro_fallspeed[t] = np.nansum(
                data["speed_classes"] * drops_per_time_and_speed
            ) / np.nansum(disdro_tr[t, :, :])

        # QC / QF

        # QC relationship v(dsd)

        def f_th(x):
            return 9.40 * (
                1 - np.exp(-1.57 * (10**3) * np.power(x * (10**-3), 1.15))
            )  # Gun and Kinzer (th.)

        y_th_disdro = f_th(data["size_classes"])

        ratio_vdisdro_vth = np.zeros(len(data.time))
        for t in range(len(data.time)):
            drops_per_time_and_diameter = np.nansum(
                disdro_tr[t, :, :], axis=1
            ).flatten()
            mu_d = drops_per_time_and_diameter / np.nansum(drops_per_time_and_diameter)
            vth_disdro = np.nansum(mu_d * y_th_disdro)
            vobs_disdro = disdro_fallspeed[t]
            ratio_vdisdro_vth[t] = vobs_disdro / vth_disdro

        QC_vdsd_t = (np.abs(ratio_vdisdro_vth - 1) <= 0.3).reshape((-1, 1))

        # QC on disdro Rain Rate data
        QC_RR_disdro = data.pr.values <= RRMAX

        # # Flag on rain accumulation
        # Accumulation_flag = disdro.

        # plt.figure()
        # plt.plot(disdro.time, ratio_vdisdro_vth, color="green")
        # plt.show(block=True)

        # Total QC/QF :
        print(QC_RR_disdro.shape, QC_vdsd_t.shape)
        Quality_matrix = np.hstack((QC_RR_disdro.reshape((-1, 1)), QC_vdsd_t))
        Quality_sum = Quality_matrix.all(axis=1)

        Quality_matrix_sum = np.flip(
            np.hstack((Quality_matrix, Quality_sum.reshape((-1, 1)))), axis=1
        )
        Quality_matrix_sum.astype(int)

        # Plot QC/QF
        fig, ax = plt.subplots(figsize=(10, 5))
        t = data.time
        ax.xaxis.set_major_formatter(formatter)
        cmap = colors.ListedColormap(["red", "green"])
        ax.pcolormesh(
            t,
            np.arange(len(Quality_matrix_sum.T)),
            Quality_matrix_sum.T,
            cmap=cmap,
            shading="nearest",
        )
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.set_yticks([q for q in range(len(Quality_matrix_sum.T))])
        ax.set_yticklabels(
            np.array(
                [
                    "QC Rain Rate",
                    "QC V(D)",
                    r"$\Pi$",
                ]
            )[::-1],
            fontsize=8,
        )
        ax.set_title("QF / QC timeseries")
        plt.savefig(
            data_dir
            + "/QL/{}_Quality_checks.png".format(start_time.strftime("%Y%m%d%T")),
            dpi=500,
            transparent=False,
            edgecolor="white",
        )
        plt.close()

        # Good / bad points

        x_t = Z_disdro[MN : -MN - 1].values
        y_t = Z_dcr_200m_resampled[MN : -MN - 1].values
        # print(x_t)
        # print(y_t)
        # f = np.where(np.isfinite(x_t))
        # print(f, x_t)
        filter = np.where((np.isfinite(x_t)) & (np.isfinite(y_t)))
        filter = filter[0]
        x_t = x_t[filter].reshape((-1, 1))
        y_t = y_t[filter].reshape((-1, 1))
        Q = Quality_sum[MN : -MN - 1]
        Q = Q[filter].reshape((-1, 1))
        Quality_matrix_filtered = Quality_matrix[MN : -MN - 1][filter]

        good = np.where(Q * 1 > 0)
        bad = np.where(Q * 1 == 0)
        print(len(good[0]) + len(bad[0]), Q.shape)

        # Assign outputs
        nb_points = len(Q)
        print("nb_points :", nb_points, "good :", len(good[0]), "bad :", len(bad[0]))

        if filtered is True:
            dZdisdro = dZdisdro[MN : -MN - 1][filter][good[0]]
            dd_tr = disdro_tr[MN : -MN - 1, :, :]
            dd_tr = dd_tr[filter, :, :]
            INTEGRATED_DSD = dd_tr[good[0], :, :].sum(axis=(0, 2))
            AVG_RAINDROP_DIAMETER = (
                INTEGRATED_DSD * data["size_classes"]
            ).sum() / INTEGRATED_DSD.sum()

        else:
            dZdisdro = dZdisdro[MN : -MN - 1][filter]
            AVG_RAINDROP_DIAMETER = np.nan
        print(dZdisdro.shape)

        if dZdisdro.shape[0] >= 1:
            print(np.count_nonzero(np.isfinite(dZdisdro)), dZdisdro.shape)
            dZmedian = np.median(dZdisdro)
            dZ_Q1 = np.quantile(dZdisdro, 0.25)
            dZ_Q3 = np.quantile(dZdisdro, 0.75)
            dZ_mean = np.mean(dZdisdro)
            dZ_minval = np.min(dZdisdro)
            dZ_maxval = np.max(dZdisdro)

        else:
            (
                dZmedian,
                dZ_Q1,
                dZ_Q3,
                dZ_mean,
                dZ_minval,
                dZ_maxval,
            ) = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

        dZ_med_quartiles = np.array(
            [
                dZmedian,
                dZ_Q1,
                dZ_Q3,
                dZ_mean,
                dZ_minval,
                dZ_maxval,
            ]
        ).reshape((-1, 1))

        RainRate_criterion_ratio = np.count_nonzero(
            Quality_matrix_filtered[:, 0]
        ) / len(Q)
        QC_vdsd_t_ratio = np.count_nonzero(Quality_matrix_filtered[:, 1]) / len(Q)
        good_points_ratio = np.count_nonzero(Q) / len(Q)
        good_points_number = np.count_nonzero(Q)

    except RuntimeError:
        return None

    return (
        start_time,
        end_time,
        nb_points,
        dZ_med_quartiles,
        RainRate_criterion_ratio,
        QC_vdsd_t_ratio,
        good_points_ratio,
        good_points_number,
        AVG_RAINDROP_DIAMETER,
    )
