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


def sel_degrade(
    list_preprocessed_files, delta_length=DELTA_LENGTH, delta_event=DELTA_EVENT
):  # lst : list of paths for DD preprocessed files
    preprocessed_ds_full = xr.concat(
        (xr.open_dataset(file) for file in list_preprocessed_files[:]), dim="time"
    )

    preprocessed_ds_full["time"] = preprocessed_ds_full.time.dt.round(freq="1T")

    # preprocessed_ds = preprocessed_ds_full.isel(
    #     {"time": np.where(preprocessed_ds_full.rain.values >= CLIC)[0]}
    # )
    preprocessed_ds = preprocessed_ds_full.isel(
        {
            "time": np.where(preprocessed_ds_full.pr.values > 0)[0]
        }  # > CLIC éventuellement, mais en allant calculer cum sur le ds full ?
    )
    time_diffs = np.diff(preprocessed_ds_full.time.values) / np.timedelta64(1, "m")
    neg_diffs = np.where(time_diffs == 0)
    print(neg_diffs, neg_diffs[0].shape)
    print(preprocessed_ds_full.time.values[neg_diffs])

    preprocessed_ds["rain_cumsum"] = np.cumsum(preprocessed_ds_full["pr"] / 60)

    t = preprocessed_ds.time

    print(t, len(t.values), len(list_preprocessed_files))

    start = []
    end = []

    start_candidate = t[0]

    for i in range(len(t) - 1):
        if t[i + 1] - t[i] >= np.timedelta64(delta_event, "m"):
            if t[i] - start_candidate >= np.timedelta64(delta_length, "m"):
                start.append(start_candidate.values)
                end.append(t[i].values)
            start_candidate = t[i + 1]

    mask = np.ones(
        (len(start), 5), dtype=int
    )  # ordre : Min temperature, min rain accumulation, max wind avg, max wind, max RR
    test_values = np.empty((len(start), 6), dtype=object)

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
            "max RR / {}mn subper (mm/h)".format(CHUNK_THICKNESS): test_values[:, 1],
            "avg RR (mm/h)": test_values[:, 2],
            "Rain accumulation > {}mm".format(MIN_CUM): mask[:, 0],
            "Max Rain Rate <= {}mm".format(RRMAX): mask[:, 1],
        }
    )

    return Events, preprocessed_ds_full


def dz_per_event(
    preprocessed_ds,
    data_dir,
    start_time,
    end_time,
    filtered=True,
):
    dcr = data_dcr_event(data_dir, start_time, end_time)
    disdro = preprocessed_ds.sel(
        {"time": slice(start_time - DELTA_DISDRO, end_time + DELTA_DISDRO)}
    )
    if (dcr is None) or (disdro is None):
        return None

    try:
        # Get data
        Z_dcr = dcr.Zh.isel(
            {"range": [0, 2, 4, 5]}
        )  # 36m, 108m, 180m, 216m (108m altitude to remove)
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
                disdro.time.values,
                Z_dcr_resampled[:, i].values,
                label="radar @ {:.0f} m".format(rng),
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
            disdro.time.values,
            np.cumsum(disdro.pr / 60),
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
            disdro["psd"].values, axes=(0, 2, 1)
        )  # besoin de transposer à Juelich !
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
                INTEGRATED_DSD * disdro["size_classes"]
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


def dz_timeseries(events, preprocessed_ds, data_dir):
    start, end = events["Start_time"].iloc[0], events["End_time"].iloc[0]
    rain_acc = events["Rain accumulation (mm)"].iloc[0]
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
    ) = dz_per_event(preprocessed_ds, data_dir, start, end, filtered=True)

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

    for i in range(1, len(events["Start_time"])):
        start, end = events["Start_time"].iloc[i], events["End_time"].iloc[i]
        print("Evenement ", i, "/", len(events["Start_time"]), start, end)
        x = dz_per_event(preprocessed_ds, data_dir, start, end, filtered=True)
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

        print(
            startend.shape,
            nb_points_per_event.shape,
            dZ.shape,
            qf_ratio.shape,
            cum.shape,
            len_episodes.shape,
            raindrop_diameter.shape,
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
    data_tosave.to_csv(data_dir + "/csv/dz_data_{}_{}.csv".format(t1, t2), header=True)

    return (
        startend,
        nb_points_per_event,
        dZ,
        qf_ratio,
        cum,
        len_episodes,
        raindrop_diameter,
    )


def dz_plot(
    startend,
    nb_points_per_event,
    dZ,
    qf_ratio,
    cum,
    len_episodes,
    raindrop_diameter,
    acc_filter=False,
    showfliers=False,
    showmeans=False,
):
    t = startend[:, 0]
    plt.figure()
    for i in range(len(dZ)):
        if cum[i] > MIN_CUM:
            plt.scatter(t[i], dZ[i, 0], color="green", alpha=qf_ratio[i, 2])
        else:
            plt.scatter(t[i], dZ[i, 0], color="red", alpha=qf_ratio[i, 2])

    plt.grid()
    plt.xlabel("Date")
    plt.ylabel(r"$\Delta$Z (dBZ)$")
    plt.show(block=False)
    print(cum)

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
        np.where((cum > MIN_CUM))[0], np.where(np.isfinite(dZ[:, 0]) * 1 == 1)[0]
    )
    f = np.intersect1d(f, np.where(qf_ratio[:, 3] >= 20))
    print(f, f.shape)
    dZ_good, t_good = dZ[f, 0], t[f]
    print(np.isfinite(dZ[39, 0]), dZ[39, 0], type(dZ[39, 0]))
    dZ_moving_avg = np.convolve(dZ_good, np.ones(N) / N, mode="valid")
    print(t_good[N - 1 :].shape, dZ_moving_avg.shape)
    (moving_avg,) = ax.plot(t_good[N - 1 :], dZ_moving_avg, color="red")
    # print(
    #     t_good, t_good.shape, dZ_moving_avg, dZ_moving_avg.shape, dZ_good, dZ_good.shape
    # )

    for k in range(len(dZ)):
        if cum[k] > MIN_CUM:
            bxp_stats = [
                {
                    "mean": dZ[k, 3],
                    "med": dZ[k, 0],
                    "q1": dZ[k, 1],
                    "q3": dZ[k, 2],
                    "fliers": [dZ[k, 4], dZ[k, 5]],
                    "whishi": np.minimum(
                        dZ[k, 2] + 1 * np.abs(dZ[k, 2] - dZ[k, 1]), dZ[k, 5]
                    ),
                    "whislo": np.maximum(
                        dZ[k, 0] - 1 * np.abs(dZ[k, 2] - dZ[k, 1]), dZ[k, 4]
                    ),
                }
            ]
            mean_shape = dict(markeredgecolor="purple", marker="_")
            med_shape = dict(markeredgecolor="red", marker="_")

            # if cum[k] < MIN_CUM:
            #     boxprops = dict(color="red")
            #     fliers_shape = dict(
            #         markerfacecolor="none",
            #         marker="o",
            #         markeredgecolor="red",
            #     )

            boxprops = dict(color="green")
            fliers_shape = dict(
                markerfacecolor="none",
                marker="o",
                markeredgecolor="green",
            )

            box = ax.bxp(
                bxp_stats,
                showmeans=showmeans,
                showfliers=showfliers,
                meanprops=mean_shape,
                # medianprops=med_shape,
                # positions=t[k],
                positions=[mpl.dates.date2num(t[k])],
                widths=[1],
                flierprops=fliers_shape,
                boxprops=boxprops,
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
    plt.xlabel("Date")
    plt.ylabel("$Z_{MIRA35} - Z_{disdrometer}$ (dBZ)")
    ax.set_ylim(bottom=-30, top=30)
    locator = mpl.dates.MonthLocator(interval=1)
    formatter = mpl.dates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%b-%d"))
    dates_axe = mpl.dates.num2date(np.array(ax.get_xticks()))
    dates_axe = [d.date() for d in dates_axe]
    plt.xticks(rotation=0, fontsize=16, fontweight="semibold")
    ax.set_yticklabels(ax.get_yticks(), fontsize=18, fontweight="semibold")
    plt.title(
        r"{} - {} Time series of MIRA35 @ Jülich CC variability ({} good events with more than {:.0f}mm of rain and 20 timesteps to compute )".format(  # noqa
            t[0].strftime("%Y/%m"),
            t[-1].strftime("%Y/%m"),
            len(np.where(cum > MIN_CUM)[0]),
            MIN_CUM,
        )
        + r"\Delta Z",
        fontsize=15,
        fontweight="semibold",
    )
    plt.savefig(
        data_dir
        + "/timeseries/timeseries_bxp_{}_{}.png".format(
            t[0].strftime("%Y%m"), t[-1].strftime("%Y%m")
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
        label="mean of median biases : {:.2f}".format(np.nanmean(biases)),
    )
    plt.xlim(left=-30, right=30)
    plt.xlabel("median $Z_{MIRA35} - Z_{disdrometer}$ (dBZ)")
    plt.ylabel("Count")
    plt.title(
        "Histogram of biases over the period {} - {}".format(
            t[0].strftime("%Y/%m"), t[-1].strftime("%Y/%m")
        )
    )
    plt.savefig(
        data_dir
        + "/timeseries/pdf_dz_{}_{}.png".format(
            t[0].strftime("%Y%m"), t[-1].strftime("%Y%m")
        ),
        dpi=500,
        transparent=False,
        edgecolor="white",
    )
    # plt.show(block=False)
    return


if __name__ == "__main__":
    # station = "lindenberg"
    station = "juelich"
    data_dir = "/home/ygrit/Documents/dcrcc_data/{}".format(station)
    lst_preprocessed_files = sorted(
        glob.glob(
            "/home/ygrit/Documents/dcrcc_data/{}/disdrometer_preprocessed/*degrade.nc".format(  # noqa
                station
            )
        )
    )[:700]
    print("{} DD preprocessed files".format(len(lst_preprocessed_files)))
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
    ) = dz_timeseries(events, preprocessed_ds, data_dir)

    dz_plot(
        startend,
        nb_points_per_event,
        dZ,
        qf_ratio,
        cum,
        len_episodes,
        raindrop_diameter,
        acc_filter=False,
    )
