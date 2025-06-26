import datetime as dt
import glob
import logging
import sys

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from create_input_files_quicklooks import data_dcr_event

lgr = logging.getLogger(__name__)

# LIST_VARIABLES = ["pr", "Zdcr", "DVdcr", "Ze_tm", "psd"] # when radar data will be
# read through the preprocessed file
LIST_VARIABLES = ["pr", "Ze_tm", "psd", "range"]

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


def sel_degrade(
    list_preprocessed_files, delta_length=DELTA_LENGTH, delta_event=DELTA_EVENT
):  # lst : list of paths for DD preprocessed files
    # col = (xr.open_dataset(file) for file in list_preprocessed_files[:])
    # for c in col:
    #     print(c.dims, c.time.values[0])

    lgr.info("Begin stacking of daily preprocessed files")
    preprocessed_ds_full = xr.concat(
        (xr.open_dataset(file)[LIST_VARIABLES] for file in list_preprocessed_files[:]),
        dim="time",
    )
    lgr.info("Stacking of daily preprocessed files : success")
    # preprocessed_ds_full["time"] = preprocessed_ds_full.time.dt.round(freq="1T")
    # not necessary anymore with perfect timesteps

    lgr.info("Select timesteps with rain records")
    preprocessed_ds = preprocessed_ds_full.isel(
        {
            "time": np.where(preprocessed_ds_full.pr.values > 0)[0]
        }  # > CLIC éventuellement, mais en allant calculer cum sur le ds full ?
    )
    lgr.info("OK")

    # time_diffs = np.diff(preprocessed_ds_full.time.values) / np.timedelta64(1, "m")
    # neg_diffs = np.where(time_diffs == 0)
    # print(neg_diffs, neg_diffs[0].shape)
    # print(preprocessed_ds_full.time.values[neg_diffs])

    preprocessed_ds["rain_cumsum"] = np.cumsum(preprocessed_ds_full["pr"] / 60)

    t = preprocessed_ds.time

    # print(t, len(t.values), len(list_preprocessed_files))

    start = []
    end = []

    start_candidate = t[0]

    for i in range(len(t) - 1):
        if t[i + 1] - t[i] >= np.timedelta64(delta_event, "m"):
            if t[i] - start_candidate >= np.timedelta64(delta_length, "m"):
                start.append(start_candidate.values)
                end.append(t[i].values)
            start_candidate = t[i + 1]

    mask = np.ones((len(start), 2), dtype=int)  # order :  min rain accumulation, max RR
    test_values = np.empty((len(start), 3), dtype=object)

    # Overall constraints/flags on weather conditions

    for k in range(len(start)):
        print(k)
        accumulation = (
            preprocessed_ds["rain_cumsum"].sel({"time": end[k]})
            - preprocessed_ds["rain_cumsum"].loc[start[k]]
            + preprocessed_ds["pr"].loc[start[k]] / 60
        )  # add 1 trigger
        test_values[k, 0] = np.round(accumulation.values, decimals=3)
        if accumulation < MIN_CUM:
            mask[k, 0] = 0

        # Condition sur le taux de pluie max : faire un découpage de l'intvl de temps
        # en tronçons de 20mn et évaluer le taux de pluie sur ces tronçons.
        # Si ok pour tous les tronçons, alors événement OK.
        time_chunks = np.arange(
            np.datetime64(start[k]),
            np.datetime64(end[k]),
            np.timedelta64(CHUNK_THICKNESS, "m"),
        )
        RR_chunks = np.zeros(len(time_chunks) - 1)
        for j in range(len(time_chunks) - 1):
            RR_chunk = (
                preprocessed_ds["pr"]
                .sel(
                    {
                        "time": slice(
                            time_chunks[j], time_chunks[j + 1] - np.timedelta64(1, "m")
                        )
                    }
                )
                .mean()
                .values
            )  # mm/h
            RR_chunks[j] = RR_chunk
        RR_chunks_max = np.max(RR_chunks)
        test_values[k, 1] = np.round(RR_chunks_max.mean(), 3)
        if RR_chunks_max > RRMAX:
            mask[k, 1] = 0

        test_values[k, 2] = np.round(
            (
                preprocessed_ds["rain_cumsum"].sel({"time": end[k]})
                - preprocessed_ds["rain_cumsum"].sel({"time": start[k]})
                + preprocessed_ds["rain_cumsum"].loc[start[k]] / 60
            )
            / ((end[k] - start[k]) / np.timedelta64(1, "h")),
            decimals=3,
        ).values

    Events = pd.DataFrame(
        {
            "Start_time": start,
            "End_time": end,
            "Rain accumulation (mm)": test_values[:, 0],
            f"max RR / {CHUNK_THICKNESS}mn subper (mm/h)": test_values[:, 1],
            "avg RR (mm/h)": test_values[:, 2],
            f"Rain accumulation > {MIN_CUM}mm": mask[:, 0],
            f"Max Rain Rate <= {RRMAX}mm": mask[:, 1],
        }
    )

    return Events, preprocessed_ds_full


def dz_per_event(
    preprocessed_ds,
    data_dir,
    start_time,
    end_time,
    gate,
    filtered=True,
):
    print("RADAR_TYPE FOR DCR FILE SELECTION : ", radar_type)
    dcr = data_dcr_event(data_dir, start_time, end_time, r_type=radar_type)
    disdro = preprocessed_ds.sel(
        {"time": slice(start_time - DELTA_DISDRO, end_time + DELTA_DISDRO)}
    )
    if (dcr is None) or (disdro is None):
        return None

    try:
        # Get data
        Z_2000m = dcr.Zh.sel({"range": slice(0, 2500)})
        Z_dcr = dcr.Zh.isel({"range": np.arange(15)})
        z_disdro = disdro.Ze_tm
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
        Z_dcr_200m_resampled = Z_dcr_resampled[
            :, int(gate)
        ]  # data @ ~215m, sampling 1 minute
        Z_dcr_resampled = Z_dcr_resampled.isel({"range": [0, 2, 6, 10]})
        if len(Z_dcr_200m_resampled) != len(Z_disdro):
            print(len(Z_dcr_200m_resampled), len(Z_disdro))
            print(
                Z_dcr_resampled.time.values[50:70],
                Z_dcr_200m_resampled.time.values[50:70],
            )
            # print(Z_dcr_200m_resampled.time.values[:], Z_disdro.time.values[:])
            Z_dcr_200m_resampled = Z_dcr_200m_resampled.sel(
                {"time": Z_disdro.time.values}
            )
            print(Z_dcr_200m_resampled.values.shape)
            Z_dcr_resampled = Z_dcr_resampled.sel({"time": Z_disdro.time.values})
            print("TIME VECTOR MODIFIED")
        # Doppler = dcr.v.isel({"range": [1, 4, 6, 8]})
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
            lgr.critical(
                "Problem : no finite reflectivity data for dcr/disdro comparison"
            )
            return None

        # QL plot 2d Z
        cmap = plt.get_cmap("rainbow").copy()
        cmap.set_under("w")

        fig, ax = plt.subplots(1, 1)
        locator = mpl.dates.AutoDateLocator()
        formatter = mpl.dates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_formatter(formatter)

        pc = ax.pcolormesh(
            pd.to_datetime(Z_2000m.time.values),
            Z_2000m.range.values,
            Z_2000m.T,
            vmin=0,
            cmap=cmap,
            shading="nearest",
        )
        ax.set_ylabel(dcr.Zh.range.attrs["long_name"])
        ax.set_title(dcr.attrs["title"])  # "94 GHz" should vary
        pos = ax.get_position()
        cb_ax = fig.add_axes([0.91, pos.y0, 0.02, pos.y1 - pos.y0])
        cb1 = fig.colorbar(pc, orientation="vertical", cax=cb_ax)
        cb1.set_label(dcr.Zh.attrs["long_name"] + " (dBZ)")
        plt.savefig(
            data_dir
            + "/QL/{}_{}_{}_Z2D.png".format(
                start_time.strftime("%Y%m%d%T"),
                preprocessed_ds.radar_source,
                preprocessed_ds.disdrometer_source,
            ),
            dpi=500,
            transparent=False,
            edgecolor="white",
        )
        plt.close()

        # QL Plot Zbasta vs Zdisdro
        fig, ax = plt.subplots()
        locator = mpl.dates.AutoDateLocator()
        formatter = mpl.dates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_formatter(formatter)
        for i in range(len(Z_dcr_resampled.range.values)):
            rng = Z_dcr_resampled.range.values[i]  # - dcr.altitude.values[0]
            # print(
            #     type(rng),
            #     type(Z_dcr_resampled.range.values[i]),
            #     type(dcr.altitude.values[0]),
            #     dcr.altitude.values.shape,
            # )
            ax.plot(
                disdro.time.values,
                Z_dcr_resampled[:, i].values,
                label=f"radar @ {rng:.0f} m",
                linewidth=1,
            )
        ax.plot(
            disdro.time.values,
            Z_disdro,
            color="green",
            label="disdrometer reflectivity",
        )
        plt.ylabel("Z [dBZ]")
        plt.xlabel("Time (UTC)")
        plt.grid()
        plt.legend()
        plt.title("{} - Disdrometer and DCR reflectivity".format(dcr.attrs["location"]))
        plt.savefig(
            data_dir
            + "/QL/{}_{}_{}_Refl.png".format(
                start_time.strftime("%Y%m%d%T"),
                preprocessed_ds.radar_source,
                preprocessed_ds.disdrometer_source,
            ),
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
            disdro.time.values,
            np.cumsum(disdro.pr / 60),
            color="green",
        )
        plt.ylabel("Rain accumulation [mm]")
        plt.xlabel("Time (UTC)")
        plt.grid()
        plt.title("{} - Disdrometer rain accumulation".format(dcr.attrs["location"]))
        plt.savefig(
            data_dir
            + "/QL/{}_{}_{}_cumsum.png".format(
                start_time.strftime("%Y%m%d%T"),
                preprocessed_ds.radar_source,
                preprocessed_ds.disdrometer_source,
            ),
            dpi=500,
            transparent=False,
            edgecolor="white",
        )
        plt.close()

        # Delta Z basta / disdro

        dZdisdro = Z_dcr_200m_resampled[:] - Z_disdro[:]
        print("shape of the DZ vector", dZdisdro.shape)

        disdro_tr = np.transpose(
            disdro["psd"].values, axes=(0, 1, 2)
        )  # bNo more need to transpose now it is done in the preprocessing !
        disdro_fallspeed = np.zeros(disdro_tr.shape[0])
        for t in range(len(disdro_fallspeed)):
            drops_per_time_and_speed = np.nansum(disdro_tr[t, :, :], axis=0).flatten()
            disdro_fallspeed[t] = np.nansum(
                disdro["speed_classes"] * drops_per_time_and_speed
            ) / np.nansum(disdro_tr[t, :, :])

        # QC / QF

        # QC relationship v(dsd)

        def f_th(x):
            return 9.40 * (
                1 - np.exp(-1.57 * (10**3) * np.power(x * (10**-3), 1.15))
            )  # Gun and Kinzer (th.)

        y_th_disdro = f_th(disdro["size_classes"])

        ratio_vdisdro_vth = np.zeros(len(disdro.time))
        for t in range(len(disdro.time)):
            drops_per_time_and_diameter = np.nansum(
                disdro_tr[t, :, :], axis=1
            ).flatten()
            mu_d = drops_per_time_and_diameter / np.nansum(drops_per_time_and_diameter)
            vth_disdro = np.nansum(mu_d * y_th_disdro)
            vobs_disdro = disdro_fallspeed[t]
            ratio_vdisdro_vth[t] = vobs_disdro / vth_disdro

        QC_vdsd_t = (np.abs(ratio_vdisdro_vth - 1) <= 0.3).reshape((-1, 1))

        # QC on disdro Rain Rate data
        QC_RR_disdro = disdro.pr.values <= RRMAX

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
        t = disdro.time
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
        ax.set_title("{} - QF / QC timeseries".format(dcr.attrs["location"]))
        plt.savefig(
            data_dir
            + "/QL/{}_{}_{}_Quality_checks.png".format(
                start_time.strftime("%Y%m%d%T"),
                preprocessed_ds.radar_source,
                preprocessed_ds.disdrometer_source,
            ),
            dpi=500,
            transparent=False,
            edgecolor="white",
        )
        plt.close()

        # Good / bad points

        x_t = Z_disdro[MN : -MN - 1].values
        y_t = Z_dcr_200m_resampled[MN : -MN - 1].values
        print(x_t.shape, y_t.shape, "len of Z vectors")
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

        Z_disdro_dcr = np.hstack((x_t[good[0]], y_t[good[0]]))

        if filtered is True:
            dZdisdro = dZdisdro[MN : -MN - 1][filter][good[0]]
            dd_tr = disdro_tr[MN : -MN - 1, :, :]
            dd_tr = dd_tr[filter, :, :]
            INTEGRATED_DSD = dd_tr[good[0], :, :].sum(axis=(0, 2))
            AVG_RAINDROP_DIAMETER = (
                INTEGRATED_DSD * disdro["size_classes"]
            ).sum() / INTEGRATED_DSD.sum()

        else:
            dZdisdro = dZdisdro[MN : -MN - 1][filter]
            AVG_RAINDROP_DIAMETER = np.nan
        print("DZ shape", dZdisdro.shape)

        if dZdisdro.shape[0] >= 1:
            print(np.count_nonzero(np.isfinite(dZdisdro)), dZdisdro.shape)
            dZmedian = np.median(dZdisdro)
            dZ_Q1 = np.quantile(dZdisdro, 0.25)
            dZ_Q3 = np.quantile(dZdisdro, 0.75)
            dZ_mean = np.mean(dZdisdro)
            dZ_minval = np.min(dZdisdro)
            dZ_maxval = np.max(dZdisdro)
            RainRate_criterion_ratio = np.count_nonzero(
                Quality_matrix_filtered[:, 0]
            ) / len(Q)
            QC_vdsd_t_ratio = np.count_nonzero(Quality_matrix_filtered[:, 1]) / len(Q)
            good_points_ratio = np.count_nonzero(Q) / len(Q)
            good_points_number = np.count_nonzero(Q)

        else:
            (
                dZmedian,
                dZ_Q1,
                dZ_Q3,
                dZ_mean,
                dZ_minval,
                dZ_maxval,
            ) = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            (
                RainRate_criterion_ratio,
                QC_vdsd_t_ratio,
                good_points_ratio,
                good_points_number,
            ) = (np.nan, np.nan, np.nan, np.nan)

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
        Z_disdro_dcr,
    )


def dz_timeseries(events, preprocessed_ds, data_dir, gate, radar_type, disdro_type):
    k = 0
    start, end = events["Start_time"].iloc[k], events["End_time"].iloc[k]
    rain_acc = events["Rain accumulation (mm)"].iloc[k]
    x = dz_per_event(preprocessed_ds, data_dir, start, end, gate, filtered=True)
    print(x is not None)
    while x is None:
        k += 1
        start, end = events["Start_time"].iloc[k], events["End_time"].iloc[k]
        rain_acc = events["Rain accumulation (mm)"].iloc[k]
        x = dz_per_event(preprocessed_ds, data_dir, start, end, gate, filtered=True)

    (
        start_time,
        end_time,
        nb_points,
        dZ_med_quartiles,
        RainRate_criterion_ratio,
        QC_vdsd_t_ratio,
        good_points_ratio,
        good_points_number,
        AVG_RAINDROP_DIAMETER,
        Z_disdro_dcr,
    ) = x

    startend = np.array([[start, end]])
    nb_points_per_event = np.array([nb_points]).reshape((1, -1))
    dZ = dZ_med_quartiles.reshape((1, -1))
    qf_ratio = np.array(
        [
            RainRate_criterion_ratio,
            QC_vdsd_t_ratio,
            good_points_ratio,
            good_points_number,
        ]
    ).reshape((1, -1))
    cum = np.array([rain_acc]).reshape((1, -1))
    delta_startend = np.array([end - start], dtype="timedelta64[ms]")[0]
    len_episodes = np.array([delta_startend / np.timedelta64(1, "m")]).reshape((-1, 1))
    raindrop_diameter = np.array([AVG_RAINDROP_DIAMETER]).reshape((-1, 1))
    Z_timesteps = Z_disdro_dcr

    for i in range(k + 1, len(events["Start_time"])):
        start, end = events["Start_time"].iloc[i], events["End_time"].iloc[i]
        print("Evenement ", i, "/", len(events["Start_time"]), start, end)
        x = dz_per_event(preprocessed_ds, data_dir, start, end, gate, filtered=True)
        if x is None:
            print("NONE")
            continue
        rain_acc = events["Rain accumulation (mm)"].iloc[i]
        (
            start_time,
            end_time,
            nb_points,
            dZ_med_quartiles,
            RainRate_criterion_ratio,
            QC_vdsd_t_ratio,
            good_points_ratio,
            good_points_number,
            AVG_RAINDROP_DIAMETER,
            Z_disdro_dcr,
        ) = x

        startend = np.append(startend, np.array([[start, end]]), axis=0)
        nb_points_per_event = np.append(
            nb_points_per_event, np.array([nb_points]).reshape((1, -1)), axis=0
        )
        dZ = np.append(dZ, dZ_med_quartiles.reshape((1, -1)), axis=0)
        qf_ratio = np.append(
            qf_ratio,
            np.array(
                [
                    RainRate_criterion_ratio,
                    QC_vdsd_t_ratio,
                    good_points_ratio,
                    good_points_number,
                ]
            ).reshape((1, -1)),
            axis=0,
        )
        cum = np.append(cum, np.array([rain_acc]).reshape((1, -1)), axis=0)
        delta_startend = np.array([end - start], dtype="timedelta64[ms]")[0]
        len_episodes = np.append(
            len_episodes,
            np.array([delta_startend / np.timedelta64(1, "m")]).reshape((-1, 1)),
            axis=0,
        )
        raindrop_diameter = np.append(
            raindrop_diameter,
            np.array([AVG_RAINDROP_DIAMETER]).reshape((-1, 1)),
            axis=0,
        )
        Z_timesteps = np.vstack((Z_timesteps, Z_disdro_dcr))

        print(
            startend.shape,
            nb_points_per_event.shape,
            dZ.shape,
            qf_ratio.shape,
            cum.shape,
            len_episodes.shape,
            raindrop_diameter.shape,
            Z_timesteps.shape,
        )

    t1, t2 = startend[0, 0].strftime("%Y%m"), startend[-1, 0].strftime("%Y%m")
    data_tosave = pd.DataFrame(
        {
            "start_time": startend[:, 0],
            "end_time": startend[:, 1],
            "episode_length": len_episodes.flatten(),
            "nb_computable_points_event": nb_points_per_event.flatten(),
            "avg_raindrop_diameter": raindrop_diameter.flatten(),
            "dz_median": dZ[:, 0],
            "dz_q1": dZ[:, 1],
            "dz_q3": dZ[:, 2],
            "dz_mean": dZ[:, 3],
            "dz_minval": dZ[:, 4],
            "dz_maxval": dZ[:, 5],
            "qf_RR": qf_ratio[:, 0],
            "qf_VD": qf_ratio[:, 1],
            "good_points_ratio": qf_ratio[:, 2],
            "good_points_number": qf_ratio[:, 3],
            "cum": cum.flatten(),
            "len_episodes": len_episodes.flatten(),
        }
    )
    data_tosave.to_csv(
        data_dir
        + f"/csv/dz_data_{radar_type}_{disdro_type}_{t1}_{t2}_gate_{int(gate)}.csv",
        header=True,
    )

    return (
        startend,
        nb_points_per_event,
        dZ,
        qf_ratio,
        cum,
        len_episodes,
        raindrop_diameter,
        Z_timesteps,
    )


def dz_plot(
    location,
    radar_source,
    disdro_source,
    radar_type,
    disdro_type,
    startend,
    nb_points_per_event,
    dZ,
    qf_ratio,
    cum,
    len_episodes,
    raindrop_diameter,
    gate,
    Z_timesteps,
    min_timesteps=30,
    acc_filter=False,
    showfliers=False,
    showmeans=False,
    showcaps=False,
):
    t = startend[:, 0]

    fig, ax = plt.subplots(figsize=((20, 6)))
    # ax.axhline(
    #     y=0, color="green", alpha=1, label="Rain accumulation > {}mm".format(MIN_CUM)
    # )
    # ax.axhline(
    #     y=0, color="red", alpha=1, label="Rain accumulation < {}mm".format(MIN_CUM)
    # )
    ax.axhline(y=0, color="blue")

    # Moving average of the bias
    N = 3  # avg(T) given by T, T-1, T-2
    f = np.intersect1d(
        np.where(cum > MIN_CUM)[0], np.where(np.isfinite(dZ[:, 0]) * 1 == 1)[0]
    )
    f = np.intersect1d(f, np.where(qf_ratio[:, 3] >= min_timesteps))
    print("Events with enough rain and timesteps : ", f.shape)
    dZ_good, t_good = dZ[f, :], t[f]
    dZ_moving_avg = np.convolve(dZ_good[:, 0], np.ones(N) / N, mode="valid")
    (moving_avg,) = ax.plot(t_good[N - 1 :], dZ_moving_avg, color="red")

    for k in range(len(dZ_good)):  # should it be in range(len(dZ_good)) ?
        bxp_stats = [
            {
                "mean": dZ_good[k, 3],
                "med": dZ_good[k, 0],
                "q1": dZ_good[k, 1],
                "q3": dZ_good[k, 2],
                "fliers": [dZ_good[k, 4], dZ_good[k, 5]],
                "whishi": np.minimum(
                    dZ_good[k, 2] + 1 * np.abs(dZ_good[k, 2] - dZ_good[k, 1]),
                    dZ_good[k, 5],
                ),
                "whislo": np.maximum(
                    dZ_good[k, 1] - 1 * np.abs(dZ_good[k, 2] - dZ_good[k, 1]),
                    dZ_good[k, 4],
                ),
            }
        ]
        mean_shape = dict(markeredgecolor="orange", marker="_")
        # med_shape = dict(markeredgecolor="red", marker="_")
        med_shape = dict(
            marker="*",
            markersize=6,
            markerfacecolor="darkorange",
            markeredgecolor="darkorange",
        )

        # if cum[k] < MIN_CUM:
        #     boxprops = dict(color="red")
        #     fliers_shape = dict(
        #         markerfacecolor="none",
        #         marker="o",
        #         markeredgecolor="red",
        #     )

        boxprops = dict(color="green")
        flierprops = dict(
            markerfacecolor="none",
            marker="o",
            markeredgecolor="green",
        )
        whiskerprops = dict(lw=0.0)

        box = ax.bxp(
            bxp_stats,
            showmeans=showmeans,
            showfliers=showfliers,
            showcaps=showcaps,
            meanprops=mean_shape,
            medianprops=med_shape,
            # positions=t[k],
            positions=[mpl.dates.date2num(t_good[k])],
            widths=[1],
            flierprops=flierprops,
            boxprops=boxprops,
            whiskerprops=whiskerprops,
        )
    if showmeans:
        ax.legend(
            [moving_avg, box["medians"][0], box["means"][0]],
            ["bias moving avg (3 values)", "median", "mean"],
            loc="upper right",
        )
    else:
        ax.legend(
            [moving_avg, box["medians"][0]],
            ["bias moving avg (3 values)", "median"],
            loc="upper right",
        )
    plt.grid()
    plt.xlabel("Date", fontsize=15, fontweight=300)
    plt.ylabel("$Z_{DCR} - Z_{disdrometer}$ (dBZ)", fontsize=15, fontweight=300)
    ax.set_ylim(bottom=-30, top=30)
    locator = mpl.dates.MonthLocator(interval=1)
    formatter = mpl.dates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%b-%d"))
    dates_axe = mpl.dates.num2date(np.array(ax.get_xticks()))
    dates_axe = [d.date() for d in dates_axe]
    plt.xticks(rotation=45, fontsize=13, fontweight="semibold")
    ax.set_yticklabels(ax.get_yticks(), fontsize=18, fontweight="semibold")
    plt.title(
        "{} - {} Time series of {} @ {} CC variability \n".format(
            t[0].strftime("%Y/%m"),
            t[-1].strftime("%Y/%m"),
            radar_source,
            location,
        )
        + r"({} good events -- more than {:.0f}mm of rain and {} timesteps to compute ".format(  # noqa
            len(f), MIN_CUM, int(min_timesteps)
        )
        + r"$\Delta Z$)",
        fontsize=13,
        fontweight="semibold",
    )
    plt.text(
        x=pd.Timestamp((t[0].value + t[-1].value) / 2.0),
        y=26,
        s="disdrometer used as a reference : " + disdro_source,
        fontsize=14,
        ha="center",
    )
    plt.text(
        x=pd.Timestamp((t[0].value + t[-1].value) / 2.0),
        y=23,
        s="No weather station data for QC",
        fontsize=14,
        ha="center",
    )
    plt.text(
        x=pd.Timestamp((t[0].value + t[-1].value) / 2.0),
        y=20,
        s=rf"Gate n° {int(gate) + 1} ({int(preprocessed_ds.range.values[int(gate)])}m AGL) used for $\Delta Z$ computation",  # noqa: E501
        fontsize=14,
        ha="center",
    )
    plt.savefig(
        data_dir
        + "/timeseries/timeseries_bxp_{}_{}_{}_{}_gate_{}.png".format(
            radar_type,
            disdro_type,
            t[0].strftime("%Y%m"),
            t[-1].strftime("%Y%m"),
            int(gate),
        ),
        dpi=500,
        transparent=False,
        edgecolor="white",
    )
    plt.show(block=False)

    plt.figure()
    # filt = np.where(cum > MIN_CUM)
    # print(filt[0])
    biases = dZ[f, 0].flatten()
    print(qf_ratio[f, 2] / 100)
    plt.hist(biases, bins=np.arange(-30, 31, 1), alpha=0.5, color="green")
    plt.axvline(
        x=np.nanmean(biases),
        color="red",
        label=f"mean of median biases : {np.nanmean(biases):.2f} dBZ",
    )
    plt.xlim(left=-30, right=30)
    plt.xlabel("median $Z_{MIRA35} - Z_{disdrometer}$ (dBZ)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid()
    plt.title(
        "Histogram of biases over the period {} - {} \n".format(
            t[0].strftime("%Y/%m"), t[-1].strftime("%Y/%m")
        )
        + f"Disdrometer : {disdro_source}, DCR : {radar_source}"
    )
    plt.savefig(
        data_dir
        + "/timeseries/pdf_dz_{}_{}_{}_{}_gate_{}.png".format(
            radar_type,
            disdro_type,
            t[0].strftime("%Y%m"),
            t[-1].strftime("%Y%m"),
            int(gate),
        ),
        dpi=500,
        transparent=False,
        edgecolor="white",
    )
    plt.close()

    # Scatter plot Z_disdrometer vs. Z_radar cumulated over the whole studied time period # noqa E501
    fig, ax = plt.subplots()
    ax.axis("equal")
    ax.set_xlim(left=-25, right=40)
    ax.set_ylim(bottom=-25, top=30)
    ax.scatter(
        Z_timesteps[:, 0],
        Z_timesteps[:, 1],
        s=2,
        c="green",
    )
    ax.plot([-25, 40], [-25, 40], color="red", label="Id")
    ax.set_title(
        "{} - {} @ {} : \n".format(
            t[0].strftime("%Y/%m"), t[-1].strftime("%Y/%m"), location
        )
        + "Scatterplot of disdrometer-based reflectivity VS DCR reflectivity \n"
        + f"Disdrometer : {disdro_source}, DCR : {radar_source}",
        fontsize=11,
        fontweight="semibold",
    )
    ax.grid()
    ax.set_xlabel(r"$Z_{disdrometer}$ [dBZ]")
    ax.set_ylabel(r"$Z_{DCR}$ [dBZ]")
    ax.legend()
    plt.savefig(
        data_dir
        + "/timeseries/scatterplot_Z_{}_{}_{}_{}_gate_{}.png".format(
            radar_type,
            disdro_type,
            t[0].strftime("%Y%m"),
            t[-1].strftime("%Y%m"),
            int(gate),
        ),
        dpi=500,
        transparent=False,
        edgecolor="white",
    )
    plt.close()

    # PDF dZ timestep by timestep, cumulated over the whole studied time period
    fig, ax = plt.subplots()
    ax.set_xlim(left=-30, right=30)
    # ax.hist(Z_timesteps[:, 1] - Z_timesteps[:, 0], color="green", alpha=0.5, bins = np.arange(-30, 31, 1)) # noqa E501
    ax.hist(
        Z_timesteps[:, 1] - Z_timesteps[:, 0],
        color="green",
        alpha=0.5,
        bins=np.arange(-30, 31, 1),
        density=True,
    )

    median_dz = np.nanmedian(Z_timesteps[:, 1] - Z_timesteps[:, 0])
    ax.axvline(
        x=median_dz,
        color="red",
        label=rf"Median $\Delta Z$ = {median_dz:.2f} dBZ",
    )
    ax.grid()
    ax.set_xlabel(r"$Z_{radar} - Z_{disdrometer}$ [dBZ]")
    ax.set_ylabel("% of values")
    # ax.set_yticklabels(np.round(100 / Z_timesteps.shape[0] * np.array(ax.get_yticks()), decimals=1)) # noqa E501
    ax.set_ylim(top=0.15)
    ax.set_yticklabels(np.round(100 * np.array(ax.get_yticks()), decimals=0))

    ax.legend()
    ax.set_title(
        "PDF of $\Delta Z$ timestep by timestep \n"
        + "Studied period : {} - {} \n".format(
            t[0].strftime("%Y/%m"), t[-1].strftime("%Y/%m")
        )
        + f"Disdrometer : {disdro_source}, DCR : {radar_source}",
        fontsize=11,
        fontweight="semibold",
    )
    plt.savefig(
        data_dir
        + "/timeseries/timestep_DZpdf_{}_{}_{}_{}_gate_{}.png".format(
            radar_type,
            disdro_type,
            t[0].strftime("%Y%m"),
            t[-1].strftime("%Y%m"),
            int(gate),
        ),
        dpi=500,
        transparent=False,
        edgecolor="white",
    )
    plt.close()

    return


if __name__ == "__main__":
    station = sys.argv[1]  # "lindenberg", "juelich", "norunda"
    radar_type = sys.argv[2]  # "rpg-fmcw-94", "mira"
    disdro_type = sys.argv[3]  # "thies-lnm", "parsivel"
    # station = str(sys.argv[1])
    # data_dir = "/home/ygrit/Documents/dcrcc_data/{}".format(station)
    data_dir = str(sys.argv[4])  # includes station name
    gate = int(sys.argv[5])
    lst_preprocessed_files = sorted(
        glob.glob(
            data_dir
            + f"/disdrometer_preprocessed/*degrade_{disdro_type}_{radar_type}.nc"
        )
    )[-246:-126]
    print(f"{len(lst_preprocessed_files)} DD preprocessed files")
    events, preprocessed_ds = sel_degrade(lst_preprocessed_files)
    print(events)

    print("#################")

    (
        startend,
        nb_points_per_event,
        dZ,
        qf_ratio,
        cum,
        len_episodes,
        raindrop_diameter,
        Z_timesteps,
    ) = dz_timeseries(events, preprocessed_ds, data_dir, gate, radar_type, disdro_type)

    ds = xr.open_dataset(lst_preprocessed_files[0])
    # print(list(ds.keys()))
    location = preprocessed_ds.location
    disdro_source = preprocessed_ds.disdrometer_source
    radar_source = preprocessed_ds.radar_source

    dz_plot(
        location,
        radar_source,
        disdro_source,
        radar_type,
        disdro_type,
        startend,
        nb_points_per_event,
        dZ,
        qf_ratio,
        cum,
        len_episodes,
        raindrop_diameter,
        gate=gate,
        Z_timesteps=Z_timesteps,
        acc_filter=False,
        showfliers=False,
        showcaps=False,
    )
