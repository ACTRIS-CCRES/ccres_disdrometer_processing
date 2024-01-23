import datetime as dt
import glob
import logging
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

lgr = logging.getLogger(__name__)

CLIC = 0.2  # mm/mn : pluvio sampling. Varies between the different weather stations ?
RRMAX = 3  # mm/h
MIN_CUM = 3  # mm/episode
MAX_MEANWS = 7
MAX_WS = 10
MIN_T = 2  # °C

DELTA_EVENT = 60
DELTA_LENGTH = 180
CHUNK_THICKNESS = 15  # mn

MN = 60
DELTA_DISDRO = dt.timedelta(minutes=MN)

CUM_REL_ERROR = 0.3  # 1, max relative error in rain accumulation measurement
FALLSPEED_REL_ERROR = (
    0.3  # 1, relative difference between theoretical and disdro fall speed
)

TIMESTAMP_THRESHOLDS = [MIN_T, MAX_WS, RRMAX, FALLSPEED_REL_ERROR]
EVENT_THRESHOLDS = [MIN_CUM, CUM_REL_ERROR]

LIST_VARIABLES = [
    "rain",
    "temp",
    "ws",
    "wd",
    "rain_sum",
    "pr",
    "Zdcr",
    "Ze_tm",
    "psd",
    "RR",
]


def sel(
    list_preprocessed_files, delta_length=DELTA_LENGTH, delta_event=DELTA_EVENT
):  # lst : list of paths for DD preprocessed files
    print("Beginning rain event selection")
    print(len(list_preprocessed_files[:]), " files")
    preprocessed_ds_full = xr.concat(
        (xr.open_dataset(file)[LIST_VARIABLES] for file in list_preprocessed_files[:]),
        dim="time",
    )
    # file = list_preprocessed_files[0]
    # preprocessed_ds_full = xr.open_dataset(file)[LIST_VARIABLES]
    # for k in range(1,len(list_preprocessed_files)):
    #     print(k, list_preprocessed_files[k])
    #     file = list_preprocessed_files[k]
    #     preprocessed_ds_full = xr.concat((preprocessed_ds_full, xr.open_dataset(file)[LIST_VARIABLES]), dim="time")

    preprocessed_ds = preprocessed_ds_full.isel(
        {"time": np.where(preprocessed_ds_full.rain.values > 0)[0]}
    )
    print(
        len(preprocessed_ds.time.values), len(np.cumsum(preprocessed_ds_full["rain"]))
    )
    print(np.where(preprocessed_ds.indexes["time"].duplicated(keep=False)))
    preprocessed_ds["rain_cumsum"] = np.cumsum(
        preprocessed_ds["rain"]
    )  # ds_full["rain"] auparavant

    t = preprocessed_ds.time

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
        min_temperature = (
            preprocessed_ds_full["temp"].sel({"time": slice(start[k], end[k])}).min()
        )
        test_values[k, 0] = np.round(min_temperature.values - 273.15, decimals=2)
        if min_temperature < MIN_T:
            mask[k, 0] = 0

        accumulation = (
            preprocessed_ds["rain_cumsum"].sel({"time": end[k]})
            - preprocessed_ds["rain_cumsum"].loc[start[k]]
            + CLIC
        )  # add 1 trigger
        test_values[k, 1] = np.round(accumulation.values, decimals=2)
        if accumulation < MIN_CUM:
            mask[k, 1] = 0

        avg_ws = (
            preprocessed_ds_full["ws"].sel({"time": slice(start[k], end[k])}).mean()
        )
        test_values[k, 2] = np.round(avg_ws.values, decimals=2)
        if avg_ws > MAX_MEANWS:
            mask[k, 2] = 0

        max_ws = preprocessed_ds_full["ws"].sel({"time": slice(start[k], end[k])}).max()
        test_values[k, 3] = np.round(max_ws.values, decimals=2)
        if max_ws > MAX_WS:
            mask[k, 3] = 0

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
                preprocessed_ds["rain"]
                .sel(
                    {
                        "time": slice(
                            time_chunks[j], time_chunks[j + 1] - np.timedelta64(1, "m")
                        )
                    }
                )
                .sum()
                .values
                * 60.0
                / CHUNK_THICKNESS
            )  # mm/h
            RR_chunks[j] = RR_chunk
        RR_chunks_max = np.max(RR_chunks)
        test_values[k, 4] = np.round(RR_chunks_max.mean(), 3)
        if RR_chunks_max > RRMAX:
            mask[k, 4] = 0

        test_values[k, 5] = np.round(
            (
                preprocessed_ds["rain_cumsum"].sel({"time": end[k]})
                - preprocessed_ds["rain_cumsum"].sel({"time": start[k]})
            )
            / ((end[k] - start[k]) / np.timedelta64(1, "h")),
            decimals=2,
        ).values

    Events = pd.DataFrame(
        {
            "Start_time": start,
            "End_time": end,
            "Min Temperature (°C)": test_values[:, 0],
            "Rain accumulation (mm)": test_values[:, 1],
            "Avg WS (m/s)": test_values[:, 2],
            "Max WS (m/s)": test_values[:, 3],
            "max RR / {}mn subper (mm/h)".format(CHUNK_THICKNESS): test_values[:, 4],
            "avg RR (mm/h)": test_values[:, 5],
            "Min Temperature < {}°C".format(MIN_T): mask[:, 0],
            "Rain accumulation > {}mm".format(MIN_CUM): mask[:, 1],
            "Avg WS < {}m/s ?".format(MAX_MEANWS): mask[:, 2],
            "Max WS < {}m/s".format(MAX_WS): mask[:, 3],
            "Max Rain Rate <= {}mm".format(RRMAX): mask[:, 4],
        }
    )
    Events.to_csv(
        data_dir
        + "/bdd_rain_events/events_{}_{}.csv".format(
            pd.Timestamp(start[0]).strftime("%Y%m"),
            pd.Timestamp(end[-1]).strftime("%Y%m"),
        )
    )

    return Events, preprocessed_ds_full


def data_event(
    preprocessed_ds,
    start_time,
    end_time,
    threshold=TIMESTAMP_THRESHOLDS + EVENT_THRESHOLDS,
    main_wind_dir=270,
):

    data_event = preprocessed_ds.sel(
        {"time": slice(start_time - DELTA_DISDRO, end_time + DELTA_DISDRO)}
    )
    if data_event.time.size == 0:
        return None

    data_event["main_wind_dir"] = main_wind_dir  # to be treated earlier
    data_event["disdro_rain_sum"] = np.cumsum(data_event["RR"]) / 60
    data_event["ws_rain_sum"] = np.cumsum(data_event["rain"])

    # Quality Flags

    data_event["QF_T"] = data_event["temp"] > threshold[0] + 273.15

    data_event["QF_ws"] = data_event["ws"] < threshold[1]

    data_event["QF_wd"] = (np.abs(data_event["wd"]) - main_wind_dir < 45) | (
        np.abs(data_event["wd"]) - (360 - main_wind_dir) < 45
    )

    data_event["QF_acc"] = data_event["rain_sum"] > threshold[4]

    data_event["QF_RR"] = xr.DataArray(
        data=np.full(len(data_event.time), True, dtype=bool), dims=["time"]
    )
    time_chunks = np.arange(
        np.datetime64(start_time),
        np.datetime64(end_time),
        np.timedelta64(CHUNK_THICKNESS, "m"),
    )
    for start_time_chunk, stop_time_chunk in zip(time_chunks[:-1], time_chunks[1:]):
        RR_chunk = (
            data_event["rain"]
            .sel(
                {
                    "time": slice(
                        start_time_chunk, stop_time_chunk - np.timedelta64(1, "m")
                    )
                }
            )
            .sum()
            * 60.0
            / CHUNK_THICKNESS
        )
        time_slice = np.where(
            (data_event.time >= start_time_chunk)
            & (data_event.time <= stop_time_chunk - np.timedelta64(1, "m"))
        )
        data_event["QF_RR"].values[time_slice] = np.tile(
            (RR_chunk <= threshold[2]), CHUNK_THICKNESS
        )

    return data_event


def dz_per_event(
    preprocessed_ds,
    data_dir,
    start_time,
    end_time,
    gate=8,  # 215m at SIRTA
    filtered=True,
):
    data = data_event(preprocessed_ds, start_time, end_time)
    if data is None:  # if len(data) != timedelta(end-start)+2 Delta ?
        return None

    try:
        # Get data
        Z_dcr = data.Zdcr.isel(
            {"range": np.arange(15)}  # [3, 4, 6, 8] good for sirta for quicklooks
        )
        z_disdro = data.Ze_tm
        z_disdro[np.where(z_disdro == 0)] = np.nan  # avoid np.inf in Z_disdro
        Z_disdro = 10 * np.log10(z_disdro)
        Z_dcr_200m = Z_dcr[:, int(gate)]  # data @ ~215m, sampling 1 minute

        # if len(Z_dcr_200m) != len(Z_disdro):
        #     Z_dcr_200m = Z_dcr_200m.sel(
        #         {"time": Z_disdro.time.values}
        #     )
        #     Z_dcr = Z_dcr.sel({"time": Z_disdro.time.values})
        #     print("TIME VECTOR MODIFIED")

        if len(np.where((np.isfinite(Z_dcr_200m)) & (np.isfinite(Z_disdro)))[0]) == 0:
            lgr.critical(
                "Problem : no finite reflectivity data for dcr/disdro comparison"
            )
            return None

        # Delta Z basta / disdro

        dZdisdro = Z_dcr_200m[:] - Z_disdro[:]

        disdro_tr = np.transpose(
            data["psd"].values, axes=(0, 1, 2)
        )  # No more need to transpose now it is done in the preprocessing !
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

        # QC relative error in rain accumulation dd/ws
        Delta_cum_pluvio_disdro = (
            data["disdro_rain_sum"][:] - data["ws_rain_sum"]
        ) / data["ws_rain_sum"]
        QC_delta_cum = (np.abs(Delta_cum_pluvio_disdro) <= 0.3).values.reshape((-1, 1))

        # Weather QFs
        QF_meteo = np.vstack(
            (
                data.QF_T.values,
                data.QF_ws.values,
                data.QF_wd.values,
                data.QF_acc.values,
                data.QF_RR.values,
            )
        ).T
        QF_meteo = QF_meteo[:, [0, 1, 2, 4, 3]]  # T, ws, wd, RR, acc

        # Overall QC/QF
        Quality_matrix = np.hstack((QF_meteo, QC_delta_cum, QC_vdsd_t))
        Quality_sum = Quality_matrix[:, [0, 1, 2, 3, 6]].all(axis=1)

        Quality_matrix_sum = np.flip(
            np.hstack((Quality_matrix, Quality_sum.reshape((-1, 1)))), axis=1
        )
        Quality_matrix_sum.astype(int)
        Quality_matrix_sum = Quality_matrix_sum[:, [0, 1, 4, 2, 3, 5, 6, 7]]

        # Good / bad points

        x_t = Z_disdro[
            MN : -MN - 1
        ].values  # sel (start, end) ? (np.Where -> same indices for Quality_matrix ?)
        y_t = Z_dcr_200m[MN : -MN - 1].values

        filter = np.where((np.isfinite(x_t)) & (np.isfinite(y_t)))
        filter = filter[0]
        x_t = x_t[filter].reshape((-1, 1))
        y_t = y_t[filter].reshape((-1, 1))
        Q = Quality_sum[MN : -MN - 1]
        Q = Q[filter].reshape((-1, 1))
        Quality_matrix_filtered = Quality_matrix[MN : -MN - 1][filter]

        good = np.where(Q * 1 > 0)
        bad = np.where(Q * 1 == 0)
        print(len(good[0]), len(bad[0]), Q.shape)

        # Assign outputs
        nb_points = len(Q)
        print("nb_points :", nb_points, "good :", len(good[0]), "bad :", len(bad[0]))

        if filtered is True:
            dZdisdro = dZdisdro[MN : -MN - 1][filter][good[0]]
            dd_tr = disdro_tr[MN : -MN - 1, :, :]
            dd_tr = dd_tr[filter, :, :]
            integrated_dsd = dd_tr[good[0], :, :].sum(axis=(0, 2))
            Avg_raindrop_diameter = (
                integrated_dsd * data["size_classes"]
            ).sum() / integrated_dsd.sum()

        else:
            dZdisdro = dZdisdro[MN : -MN - 1][filter]
            Avg_raindrop_diameter = np.nan

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

        Temp_criterion_ratio = np.count_nonzero(Quality_matrix_filtered[:, 0]) / len(Q)
        Wind_criterion_ratio = np.count_nonzero(Quality_matrix_filtered[:, 1]) / len(Q)
        Wind_dir_criterion_ratio = np.count_nonzero(
            Quality_matrix_filtered[:, 2]
        ) / len(Q)
        RainRate_criterion_ratio = np.count_nonzero(
            Quality_matrix_filtered[:, 3]
        ) / len(Q)
        QC_vdsd_t_ratio = np.count_nonzero(Quality_matrix_filtered[:, 6]) / len(Q)
        Accumulation_flag = Quality_matrix[-1, 4]
        Accumulation_relative_error_flag = Quality_matrix_filtered[-1, 5]
        good_points_ratio = np.count_nonzero(Q) / len(Q)
        good_points_number = np.count_nonzero(Q)

    except RuntimeError:
        return None

    return (
        start_time,
        end_time,
        nb_points,
        dZ_med_quartiles,
        Temp_criterion_ratio,
        Wind_criterion_ratio,
        Wind_dir_criterion_ratio,
        RainRate_criterion_ratio,
        QC_vdsd_t_ratio,
        Accumulation_flag,
        Accumulation_relative_error_flag,
        good_points_ratio,
        good_points_number,
        Avg_raindrop_diameter,
    )


def dz_timeseries(events, preprocessed_ds, data_dir, gate):
    k = 0
    start, end = events["Start_time"].iloc[k], events["End_time"].iloc[k]
    rain_acc = events["Rain accumulation (mm)"].iloc[k]
    x = dz_per_event(preprocessed_ds, data_dir, start, end, gate=gate, filtered=True)
    print(not (x is None))
    while x is None:
        k += 1
        start, end = events["Start_time"].iloc[k], events["End_time"].iloc[k]
        rain_acc = events["Rain accumulation (mm)"].iloc[k]
        x = dz_per_event(
            preprocessed_ds, data_dir, start, end, gate=gate, filtered=True
        )

    (
        start_time,
        end_time,
        nb_points,
        dZ_med_quartiles,
        Temp_criterion_ratio,
        Wind_criterion_ratio,
        Wind_dir_criterion_ratio,
        RainRate_criterion_ratio,
        QC_vdsd_t_ratio,
        Accumulation_flag,
        Accumulation_relative_error_flag,
        good_points_ratio,
        good_points_number,
        Avg_raindrop_diameter,
    ) = x

    startend = np.array([[start, end]])
    nb_points_per_event = np.array([nb_points]).reshape((1, -1))
    dZ = dZ_med_quartiles.reshape((1, -1))
    qf_ratio = np.array(
        [
            Temp_criterion_ratio,
            Wind_criterion_ratio,
            Wind_dir_criterion_ratio,
            RainRate_criterion_ratio,
            QC_vdsd_t_ratio,
            good_points_ratio,
            good_points_number,
        ]
    ).reshape((1, -1))
    accumulation_flags = np.array([Accumulation_flag]).reshape((1, -1))
    accumulation_errors_flags = np.array([Accumulation_relative_error_flag]).reshape(
        (1, -1)
    )
    cum = np.array([rain_acc]).reshape((1, -1))
    delta_startend = np.array([end - start], dtype="timedelta64[ms]")[0]
    len_episodes = np.array([delta_startend / np.timedelta64(1, "m")]).reshape((-1, 1))
    raindrop_diameter = np.array([Avg_raindrop_diameter]).reshape((-1, 1))

    for i in range(k + 1, len(events["Start_time"])):
        start, end = events["Start_time"].iloc[i], events["End_time"].iloc[i]
        print("Evenement ", i, "/", len(events["Start_time"]), start, end)
        x = dz_per_event(
            preprocessed_ds, data_dir, start, end, gate=gate, filtered=True
        )
        if x is None:
            print("NONE")
            continue
        rain_acc = events["Rain accumulation (mm)"].iloc[i]
        (
            start_time,
            end_time,
            nb_points,
            dZ_med_quartiles,
            Temp_criterion_ratio,
            Wind_criterion_ratio,
            Wind_dir_criterion_ratio,
            RainRate_criterion_ratio,
            QC_vdsd_t_ratio,
            Accumulation_flag,
            Accumulation_relative_error_flag,
            good_points_ratio,
            good_points_number,
            Avg_raindrop_diameter,
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
                    Temp_criterion_ratio,
                    Wind_criterion_ratio,
                    Wind_dir_criterion_ratio,
                    RainRate_criterion_ratio,
                    QC_vdsd_t_ratio,
                    good_points_ratio,
                    good_points_number,
                ]
            ).reshape((1, -1)),
            axis=0,
        )
        accumulation_flags = np.append(
            accumulation_flags, np.array([Accumulation_flag]).reshape((1, -1)), axis=0
        )
        accumulation_errors_flags = np.append(
            accumulation_errors_flags,
            np.array([Accumulation_relative_error_flag]).reshape((1, -1)),
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
            np.array([Avg_raindrop_diameter]).reshape((-1, 1)),
            axis=0,
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
            "qf_temp": qf_ratio[:, 0],
            "qf_wind_speed": qf_ratio[:, 1],
            "qf_wind_dir": qf_ratio[:, 2],
            "qf_RR": qf_ratio[:, 3],
            "qf_VD": qf_ratio[:, 4],
            "good_points_ratio": qf_ratio[:, 5],
            "good_points_number": qf_ratio[:, 6],
            "enough_rain": accumulation_flags.flatten(),
            "low_acc_relative_error": accumulation_errors_flags.flatten(),
            "cum": cum.flatten(),
        }
    )
    data_tosave.to_csv(
        data_dir + "/csv/dz_data_{}_{}_gate{}.csv".format(t1, t2, int(gate)),
        header=True,
    )

    return (
        startend,
        nb_points_per_event,
        dZ,
        qf_ratio,
        accumulation_flags,
        accumulation_errors_flags,
        cum,
        len_episodes,
        raindrop_diameter,
    )


def dz_plot(
    preprocessed_ds,
    data_dir,
    startend,
    nb_points_per_event,
    dZ,
    qf_ratio,
    cum,
    accumulation_errors_flags,
    len_episodes,
    raindrop_diameter,
    gate,
    min_timesteps=30,
    acc_filter=False,
    showfliers=False,
    showmeans=False,
    showcaps=False,
    showwhiskers=False,
):
    location = preprocessed_ds.location
    disdro_source = preprocessed_ds.disdrometer_source
    radar_source = preprocessed_ds.radar_source
    t = startend[:, 0]

    fig, ax = plt.subplots(figsize=((20, 6)))

    ax.axhline(y=0, color="blue")

    # Filtering events with "enough" rain and data points to have robust dZ stats
    f = np.intersect1d(
        np.where((cum > MIN_CUM))[0], np.where(np.isfinite(dZ[:, 0]) * 1 == 1)[0]
    )
    f = np.intersect1d(f, np.where(accumulation_errors_flags * 1 == 1))
    f = np.intersect1d(f, np.where(qf_ratio[:, 6] >= min_timesteps))
    print("Events with enough rain and timesteps : ", f.shape)
    dZ_good, t_good = dZ[f, :], t[f]

    # Moving average of the bias
    N = 3  # avg(T) given by T, T-1, T-2
    dZ_moving_avg = np.convolve(dZ_good[:, 0], np.ones(N) / N, mode="valid")
    (moving_avg,) = ax.plot(t_good[N - 1 :], dZ_moving_avg, color="red")

    print("GOOD : ", dZ_good.shape, dZ_moving_avg.shape, t_good[N - 1 :].shape)
    print(dZ_good[:, 0], t_good, dZ_moving_avg)

    # Boxplot props
    mean_shape = dict(markeredgecolor="purple", marker="_")
    # med_shape = dict(markeredgecolor="red", marker="_")
    med_shape = dict(linewidth=2)
    boxprops = dict(color="green")
    flierprops = dict(
        markerfacecolor="none",
        marker="o",
        markeredgecolor="green",
    )
    whiskerprops = dict(lw=1.0 * showwhiskers)

    for k in range(len(dZ_good)):
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
    plt.xlabel("Date")
    plt.ylabel("$Z_{DCR} - Z_{disdrometer}$ (dBZ)")
    ax.set_ylim(bottom=-30, top=30)

    locator = mpl.dates.MonthLocator(interval=1)
    formatter = mpl.dates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    # ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%b-%d"))

    dates_axe = mpl.dates.num2date(np.array(ax.get_xticks()))
    dates_axe = [d.date() for d in dates_axe]
    plt.xticks(rotation=45, fontsize=10, fontweight="semibold")
    ax.set_yticklabels(ax.get_yticks(), fontsize=18, fontweight="semibold")
    plt.title(
        "{} - {} Time series of {} @ {} CC variability \n ".format(
            t[0].strftime("%Y/%m"),
            t[-1].strftime("%Y/%m"),
            radar_source,
            location,
        )
        + r"{} good events -- max. {:.0f}% error between WS and DD accumulation, more than {:.0f}mm of rain and {} timesteps to compute $\Delta Z$ ".format(  # noqa
            len(f),
            CUM_REL_ERROR * 100,
            MIN_CUM,
            min_timesteps,
        ),
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
        s="Weather data used for complementary QC",
        fontsize=14,
        ha="center",
    )
    plt.text(
        x=pd.Timestamp((t[0].value + t[-1].value) / 2.0),
        y=20,
        s=r"Gate n° {} ({}m AGL) used for $\Delta Z$ computation".format(
            int(gate) + 1, int(preprocessed_ds.range.values[int(gate)])
        ),
        fontsize=14,
        ha="center",
    )
    plt.savefig(
        data_dir
        + "/timeseries/timeseries_bxp_{}_{}_gate{}.png".format(
            t[0].strftime("%Y%m"), t[-1].strftime("%Y%m"), int(gate)
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
        label="mean of median biases : {:.2f} dBZ".format(np.nanmean(biases)),
    )
    plt.xlim(left=-30, right=30)
    plt.xlabel("median $Z_{DCR} - Z_{disdrometer}$ (dBZ)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid()
    plt.title(
        "DCRCC of {} at {}, based on {} disdrometer \n".format(
            radar_source, location, disdro_source
        )
        + "Histogram of biases (at gate {}) over the period {} - {}".format(
            int(gate) + 1, t[0].strftime("%Y/%m"), t[-1].strftime("%Y/%m")
        )
    )
    plt.savefig(
        data_dir
        + "/timeseries/pdf_dz_{}_{}_gate{}.png".format(
            t[0].strftime("%Y%m"), t[-1].strftime("%Y%m"), int(gate)
        ),
        dpi=500,
        transparent=False,
        edgecolor="white",
    )
    plt.close()
    return


def dd_ql(preprocessed_ds, events):
    # for start, end in zip(events["Start_time"], events["End_time"]):
    pass


def main(data_dir, gate):
    lst = sorted(glob.glob(data_dir + "/disdrometer_preprocessed/*_preprocessed.nc"))[
        :-500
    ]
    events, preprocessed_ds = sel(lst)
    print(events)

    print("################")

    (
        startend,
        nb_points_per_event,
        dZ,
        qf_ratio,
        accumulation_flags,
        accumulation_errors_flags,
        cum,
        len_episodes,
        raindrop_diameter,
    ) = dz_timeseries(events, preprocessed_ds, data_dir, gate=gate)

    dz_plot(
        preprocessed_ds,
        data_dir,
        startend,
        nb_points_per_event,
        dZ,
        qf_ratio,
        cum,
        accumulation_errors_flags,
        len_episodes,
        raindrop_diameter,
        gate,
        acc_filter=False,
        showfliers=False,
        showmeans=False,
        showcaps=False,
        showwhiskers=False,
    )

    return True


if __name__ == "__main__":
    station = str(sys.argv[1])
    # station = "palaiseau"
    data_dir = "/homedata/ygrit/disdro_processing/" + station
    gate = sys.argv[2]
    output = main(data_dir, gate)
    print(output)
