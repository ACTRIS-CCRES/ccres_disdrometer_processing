# (Obsolete)
# Compute statistics for reflectivity differences during one rain event,
# with or without applying filtering options

import datetime as dt
import logging

import numpy as np
import pandas as pd

from . import create_input_files_quicklooks as create_files

lgr = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None  # default='warn'

CHUNK_THICKNESS = 15  # minutes

MN = 60  # mn : Time horizon before and after therain event
MAX_WS = 7  # m/s : Wind speed threshold for QC/QF
MIN_T = 2  # °C : Min temperature threshold for QC/QF
MIN_CUM = 3  # mm : minimum rain accumulation to keep an event in the statistics
MAX_RR = 3

CUM_REL_ERROR = 0.3  # 1, max rel error in rain accumulation btwn disdro and pluvio
FALLSPEED_REL_ERROR = (
    0.3  # 1, relative difference btwn theoretical and disdro fall speed
)

TIMESTAMP_THRESHOLDS = [MIN_T, MAX_WS, MAX_RR, FALLSPEED_REL_ERROR]
EVENT_THRESHOLDS = [MIN_CUM, CUM_REL_ERROR]

DELTA_DISDRO = dt.timedelta(minutes=MN)


def dz_per_event(
    start_time,
    end_time,
    thresholds=TIMESTAMP_THRESHOLDS + EVENT_THRESHOLDS,
    filtered=True,
):
    weather, dcr, disdro = create_files.get_data_event(start_time, end_time, thresholds)

    if (weather is None) or (dcr is None) or (disdro is None):
        return None

    try:
        # Get data
        Z_dcr = dcr.Zh.isel({"range": [1, 4, 6, 8]})
        z_disdro = disdro.Ze_tm
        z_disdro[np.where(z_disdro == 0)] = np.nan  # avoid np.inf in Z_disdro
        Z_disdro = 10 * np.log10(z_disdro)
        time_index = pd.date_range(
            start_time, end_time + pd.Timedelta(minutes=1), freq="1T"
        )
        time_index_offset = time_index - pd.Timedelta(30, "sec")

        Z_dcr_resampled = Z_dcr.groupby_bins(
            "time", time_index_offset, labels=time_index[:-1]
        ).median(dim="time", keep_attrs=True)
        Z_dcr_resampled = Z_dcr_resampled.rename({"time_bins": "time"})
        Z_dcr_200m_resampled = Z_dcr_resampled[:, 3]  # data @ 212.5m, sampling 1 minute
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

        # Delta Z basta / disdro

        dZdisdro = Z_dcr_200m_resampled[:] - Z_disdro[:]

        disdro_tr = np.transpose(disdro["psd"].values, axes=(0, 2, 1))
        disdro_fallspeed = np.zeros(disdro_tr.shape[0])
        for t in range(len(disdro_fallspeed)):
            drops_per_time_and_speed = np.sum(disdro_tr[t, :, :], axis=0).flatten()
            disdro_fallspeed[t] = np.sum(
                disdro["speed_classes"] * drops_per_time_and_speed
            ) / np.sum(disdro_tr[t, :, :])

        # QC / QF

        # Rain accumulation differences

        Delta_cum_pluvio_disdro = (disdro["cum"][:] - weather["cum"]) / weather["cum"]

        # Plot relationship v(dsd) and compare it to theoretical curve

        def f_th(x):
            return 9.40 * (
                1 - np.exp(-1.57 * (10**3) * np.power(x * (10**-3), 1.15))
            )  # Gun and Kinzer (th.)

        y_th_disdro = f_th(disdro["size_classes"])

        ratio_vdisdro_vth = np.zeros(len(disdro.time))
        for t in range(len(disdro.time)):
            drops_per_time_and_diameter = np.sum(disdro_tr[t, :, :], axis=1).flatten()
            mu_d = drops_per_time_and_diameter / np.sum(drops_per_time_and_diameter)
            vth_disdro = np.sum(mu_d * y_th_disdro)
            vobs_disdro = disdro_fallspeed[t]
            ratio_vdisdro_vth[t] = vobs_disdro / vth_disdro

        # Meteo, matrice QC

        QC_delta_cum = (np.abs(Delta_cum_pluvio_disdro) <= 0.3).values.reshape((-1, 1))
        QF_meteo = np.vstack(
            (
                weather.QF_T.values,
                weather.QF_ws.values,
                weather.QF_acc.values,
                weather.QF_RR.values,
            )
        ).T
        QF_meteo = QF_meteo[:, [0, 1, 3, 2]]
        QC_vdsd_t = (np.abs(ratio_vdisdro_vth - 1) <= 0.3).reshape((-1, 1))
        # maybe here add later QC on disdro fall speed or wind direction

        Quality_matrix = np.hstack((QF_meteo, QC_delta_cum, QC_vdsd_t))
        Quality_sum = Quality_matrix[:, [0, 1, 2, 5]].all(axis=1)

        Quality_matrix_sum = np.flip(
            np.hstack((Quality_matrix, Quality_sum.reshape((-1, 1)))), axis=1
        )
        Quality_matrix_sum.astype(int)

        # Good / bad points

        x_t = Z_disdro[MN : -MN - 1].values
        y_t = Z_dcr_200m_resampled[MN : -MN - 1].values
        filter = np.where(
            (np.isfinite(x_t) is True)
            & (np.isfinite(y_t) is True)
            & (np.isnan(x_t) is False)
            & (np.isnan(y_t) is False)
        )[0]
        x_t = x_t[filter].reshape((-1, 1))
        y_t = y_t[filter].reshape((-1, 1))
        Q = Quality_sum[MN : -MN - 1]
        Q = Q[filter].reshape((-1, 1))
        Quality_matrix_filtered = Quality_matrix[MN : -MN - 1][filter]

        good = np.where(Q is True)
        bad = np.where(Q is False)

        # Assign outputs
        nb_points = len(Q)
        print("nb_points :", nb_points, "good :", len(good[0]), "bad :", len(bad[0]))

        if filtered is True:
            dZdisdro = dZdisdro[MN : -MN - 1][filter][good[0]]
            INTEGRATED_DSD = disdro_tr[good[0], :, :].sum(axis=(0, 2))
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

        Temp_criterion_ratio = np.count_nonzero(Quality_matrix_filtered[:, 0]) / len(Q)
        Wind_criterion_ratio = np.count_nonzero(Quality_matrix_filtered[:, 1]) / len(Q)
        RainRate_criterion_ratio = np.count_nonzero(
            Quality_matrix_filtered[:, 2]
        ) / len(Q)
        Accumulation_flag = Quality_matrix[-1, 3]
        Accumulation_relative_error_flag = Quality_matrix_filtered[-1, 4]
        QC_vdsd_t_ratio = np.count_nonzero(Quality_matrix_filtered[:, 5]) / len(Q)
        # doppler_speed_good_ratio = 0
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
        RainRate_criterion_ratio,
        QC_vdsd_t_ratio,
        Accumulation_flag,
        Accumulation_relative_error_flag,
        good_points_ratio,
        good_points_number,
        AVG_RAINDROP_DIAMETER,
    )
