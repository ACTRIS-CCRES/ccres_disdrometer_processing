"""Apply the processing from daily preprocessed files.

Input : Daily preprocessed files at days D and D-1
Output : Daily processed file for day D
"""

import logging

import numpy as np
import pandas as pd
import xarray as xr

lgr = logging.getLogger(__name__)

QC_FILL_VALUE = -9


def rain_event_selection_weather(ds, conf):  # with no constraint on cum for the moment
    sel_ds = ds.isel(
        {
            "time": np.where(
                ds.ams_pr.values / 60
                >= conf["instrument_parameters"]["RAIN_GAUGE_SAMPLING"]
            )[0]
        }
    )  # noqa

    min_duration, max_interval = (
        conf["thresholds"]["MIN_DURATION"],
        conf["thresholds"]["MAX_INTERVAL"],
    )

    t = sel_ds.time
    start, end = [], []
    start_candidate = t[0]
    for i in range(len(t) - 1):
        if i + 1 == len(t) - 1:
            if t[i + 1] - t[i] > np.timedelta64(max_interval, "m"):
                if t[i] - start_candidate >= np.timedelta64(min_duration, "m"):
                    start.append(start_candidate.values)
                    end.append(t[i].values)
            elif t[i + 1] - start_candidate >= np.timedelta64(min_duration, "m"):
                start.append(start_candidate.values)
                end.append(t[i + 1].values)
        if t[i + 1] - t[i] > np.timedelta64(max_interval, "m"):
            if t[i] - start_candidate >= np.timedelta64(min_duration, "m"):
                start.append(start_candidate.values)
                end.append(t[i].values)
            start_candidate = t[i + 1]
    return start, end


def compute_quality_checks_weather(ds, conf, start, end):
    qc_ds = xr.Dataset(coords=dict(time=(["time"], ds.time.data)))

    # flag the timesteps belonging to an event
    qc_ds["flag_event"] = xr.DataArray(
        data=np.full(len(ds.time), False, dtype=bool),
        dims=["time"],
        attrs={
            "long_name": "flag to describe if a timestep belongs to a rain event",
            "unit": "1",
        },
    )
    # do a column for rain accumulation since last beginning of an event
    qc_ds["ams_cp_since_event_begin"] = xr.DataArray(
        np.nan * np.zeros(len(ds.time)),
        dims=["time"],
        attrs={
            "long_name": "pluviometer rain accumulation since last begin of an event",
            "unit": "mm",
            "comment": "not computable when no AMS data is provided",
        },
    )
    qc_ds["disdro_cp_since_event_begin"] = xr.DataArray(
        np.zeros(len(ds.time)),
        dims="time",
        attrs={
            "long_name": "disdrometer rain accumulation since last begin of an event",
            "unit": "mm",
        },
    )
    for s, e in zip(start, end):
        # mask = (qc_ds.time >= s) & (qc_ds.time <= e)
        # qc_ds["ams_cp_since_event_begin"] = qc_ds["ams_cp_since_event_begin"].where(
        #     ~mask, 1
        # )
        qc_ds["flag_event"].loc[slice(s, e)] = True

        qc_ds["ams_cp_since_event_begin"].loc[slice(s, e)] = (
            1 / 60 * np.nancumsum(ds["ams_pr"].sel(time=slice(s, e)).values)
        )
        qc_ds["disdro_cp_since_event_begin"].loc[slice(s, e)] = (
            1 / 60 * np.nancumsum(ds["disdro_pr"].sel(time=slice(s, e)).values)
        )

    # Flag for condition (rainfall_amount > N mm)
    qc_ds["QF_rainfall_amount"] = xr.DataArray(
        qc_ds["ams_cp_since_event_begin"] >= conf["thresholds"]["MIN_RAINFALL_AMOUNT"],
        dims="time",
        attrs={
            "long_name": "Quality flag for minimum rainfall amount",
            "unit": "1",
            "comment": f"threshold = {conf['thresholds']['MIN_RAINFALL_AMOUNT']:.0f} mm",  # noqa E501
        },
    )

    # QC on AMS precipitation rate
    qc_ds["QC_pr"] = xr.DataArray(
        data=np.full(len(ds.time), True, dtype=bool),
        dims=["time"],
        attrs={
            "long_name": "Quality check for rainfall rate",
            "unit": "1",
            "comment": f"threshold = {conf['thresholds']['MAX_RR']:.0f} mm/h",
        },
    )

    for s, e in zip(start, end):
        time_chunks = np.arange(
            np.datetime64(s),
            np.datetime64(e),
            np.timedelta64(conf["thresholds"]["PR_SAMPLING"], "m"),
        )
        for start_time_chunk, stop_time_chunk in zip(time_chunks[:-1], time_chunks[1:]):
            RR_chunk = (
                ds["ams_pr"]
                .sel(
                    {
                        "time": slice(
                            start_time_chunk, stop_time_chunk - np.timedelta64(1, "m")
                        )
                    }
                )
                .mean()
            )
            qc_ds["QC_pr"].loc[
                slice(start_time_chunk, stop_time_chunk - np.timedelta64(1, "m"))
            ] = RR_chunk <= conf["thresholds"]["MAX_RR"]

    # QC relationship v(dsd)
    vth_disdro = (
        np.nansum(ds["psd"].values, axis=2) @ ds["modV"].isel(time=0).values
    ) / np.nansum(
        ds["psd"].values, axis=(1, 2)
    )  # average fall speed weighed by (num_drops_per_time_and_diameter)
    vobs_disdro = (
        np.nansum(ds["psd"].values, axis=1) @ ds["speed_classes"].values
    ) / np.nansum(ds["psd"].values, axis=(1, 2))
    ratio_vdisdro_vth = vobs_disdro / vth_disdro

    qc_ds["QC_vdsd_t"] = xr.DataArray(
        data=(np.abs(ratio_vdisdro_vth - 1) <= 0.3),
        dims="time",
        attrs={
            "long_name": "Quality check for cohence between fall speed and droplet diameter",  # noqa E501
            "unit": "1",
            "comment": f"threshold = {conf['thresholds']['DD_FALLSPEED_RATIO']:.1f} i.e. {100*conf['thresholds']['DD_FALLSPEED_RATIO']:.0f}% of relative error between average fall speed computed from speed distribution and average fall speed modeled from diameter distribution",  # noqa E501,
        },
    )

    # Temperature QC
    qc_ds["QC_ta"] = xr.DataArray(
        ds["ta"].values > conf["thresholds"]["MIN_TEMP"],
        dims="time",
        attrs={"long_name": "Quality check for air temperature"},
    )
    # Wind speed and direction QCs
    qc_ds["QC_ws"] = xr.DataArray(
        ds["ws"].values < conf["thresholds"]["MAX_WS"],
        dims="time",
        attrs={"long_name": "Quality check for wind speed"},
    )
    main_wind_dir = (conf["instrument_parameters"]["DD_ORIENTATION"] + 90) % 360
    dd_angle = conf["thresholds"]["DD_ANGLE"]
    qc_ds["QC_wd"] = xr.DataArray(
        (np.abs(ds["wd"] - main_wind_dir) < dd_angle)
        | (np.abs(ds["wd"] - main_wind_dir) > 360 - dd_angle)
        | (np.abs(ds["wd"] - (main_wind_dir + 180) % 360) < dd_angle)
        | (np.abs(ds["wd"] - (main_wind_dir + 180) % 360) > 360 - dd_angle),
        dims="time",
        attrs={"long_name": "Quality check for wind direction"},
    )  # data is between 0 and 360°

    # Check agreement between rain gauge and disdrometer rain measurements
    # extract ds(start to end), compute relative deviation and compare to conf tolerance
    qc_ds["QF_rg_dd"] = xr.DataArray(
        data=np.full(len(ds.time), False, dtype=bool),
        dims="time",
        attrs={
            "long_name": "Quality flag for discrepancy between rain gauge and disdrometer precipitation rate"  # noqa
        },
    )
    for s, e in zip(start, end):
        # event_mask = np.where((ds.time.values >= s) | (ds.time.values <= e))[0]
        qc_ds["QF_rg_dd"].loc[slice(s, e)] = (
            np.abs(
                qc_ds["disdro_cp_since_event_begin"].loc[slice(s, e)]
                - qc_ds["ams_cp_since_event_begin"].loc[slice(s, e)]
            )
            / qc_ds["ams_cp_since_event_begin"].loc[slice(s, e)]
            < conf["thresholds"]["DD_RG_MAX_PR_ACC_RATIO"]
        )

    # attributes for weather-related QCs
    for key in ["QC_ta", "QC_wd", "QC_ws", "QF_rg_dd"]:
        qc_ds[key].attrs["unit"] = "1"
        qc_ds[key].attrs["comment"] = "not computable when no AMS data is provided"

        # Overall QC : ta, ws, wd, ams_pr, v(d)
        qc_ds["QC_overall"] = xr.DataArray(
            data=qc_ds["QC_ta"]
            & qc_ds["QC_ws"]
            & qc_ds["QC_wd"]
            & qc_ds["QC_pr"]
            & qc_ds["QC_vdsd_t"],
            dims="time",
            attrs={
                "long_name": "Overall quality check",
                "unit": "1",
                "comment": "Checks combined : vdsd_t, pr",
            },
        )

    # Data attributes and types
    for key in [
        "flag_event",
        "QF_rainfall_amount",
        "QC_ta",
        "QC_ws",
        "QC_wd",
        "QF_rg_dd",
        "QC_pr",
        "QC_vdsd_t",
        "QC_overall",
    ]:
        qc_ds[key].data.astype("i2")
        qc_ds[key].attrs["flag_values"] = np.array([0, 1]).astype("i2")

    qc_ds["flag_event"].attrs[
        "flag_meanings"
    ] = "timestep_part_of_an_event timestep_not_involved_in_any_avent"
    qc_ds["QF_rainfall_amount"].attrs[
        "flag_meanings"
    ] = "less_rain_than_threshold_since_event_begin more_rain_than_threshold_since_event_begin"  # noqa E501
    qc_ds["QC_ta"].attrs[
        "flag_meanings"
    ] = "temperature_lower_than_threshold temperature_ok"
    qc_ds["QC_ws"].attrs[
        "flag_meanings"
    ] = "wind_speed_higher_than_threshold wind_speed_ok"
    qc_ds["QC_wd"].attrs[
        "flag_meanings"
    ] = "wind_direction_outside_good_angle_range wind_direction_ok"
    qc_ds["QF_rg_dd"].attrs[
        "flag_meanings"
    ] = "relative_difference_higher_than_threshold relative_difference_ok"
    qc_ds["QC_pr"].attrs[
        "flag_meanings"
    ] = "precipitation_rate_above threshold precipitation_rate_ok"
    qc_ds["QC_vdsd_t"].attrs[
        "flag_meanings"
    ] = "discrepancy_between_observed_and_modeled_disdrometer_droplet_fallspeed_above_threshold discrepancy_under_threshold"  # noqa e501
    qc_ds["QC_overall"].attrs["flag_meanings"] = "at_least_one_QC_not_OK all_QC_OK"

    return qc_ds


def compute_todays_events_stats_weather(ds, Ze_ds, conf, qc_ds, start, end):
    n = 0
    for s in start:
        if (
            pd.to_datetime(s).day
            == pd.to_datetime(ds.time.isel(time=len(qc_ds.time) // 2).values).day
        ):
            n += 1
    # n is the number of events to store in the dataset
    # i.e. the number of events which begin at day D

    stats_ds = xr.Dataset(coords=dict(events=(["events"], np.linspace(1, 1 + n, n))))

    dZ_mean, dZ_med, dZ_q1, dZ_q3, dZ_min, dZ_max = (
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
    )
    event_length, rain_accumulation, nb_dz_computable_pts = (
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
    )
    (
        qc_ta_ratio,
        qc_ws_ratio,
        qc_wd_ratio,
        qc_vdsd_t_ratio,
        qc_pr_ratio,  # noqa # TODO
        qc_overall_ratio,
    ) = (
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
    )
    start_event, end_event = np.zeros(n), np.zeros(n)

    event = 0
    for s, e in zip(start, end):
        if (
            pd.to_datetime(s).day
            == pd.to_datetime(ds.time.isel(time=len(qc_ds.time) // 2).values).day
        ):
            start_event[event] = s
            end_event[event] = e
            r = conf["instrument_parameters"]["DCR_DZ_RANGE"]
            dz_r = Ze_ds["Delta_Z"].sel(time=slice(s, e)).sel(range=r, method="nearest")
            dz_r_nonan = dz_r[np.isfinite(dz_r)]
            # General info about the event
            event_length[event] = (e - s) / np.timedelta64(1, "m") + 1
            rain_accumulation[event] = qc_ds["ams_cum_since_event_begin"].loc[e]
            nb_dz_computable_pts[event] = len(dz_r)
            # QC passed ratios
            qc_ds_event = qc_ds.sel(time=slice(s, e)).loc[{"time": np.isfinite(dz_r)}]
            print(qc_ds_event.QC_ta.values.shape)
            qc_ta_ratio[event] = np.sum(qc_ds_event["QC_ta"]) / len(qc_ds_event.time)
            qc_ws_ratio[event] = np.sum(qc_ds_event["QC_ws"]) / len(qc_ds_event.time)
            qc_wd_ratio[event] = np.sum(qc_ds_event["QC_wd"]) / len(qc_ds_event.time)
            qc_vdsd_t_ratio[event] = np.sum(qc_ds_event["QC_vdsd_t"]) / len(
                qc_ds_event.time
            )
            qc_overall_ratio[event] = np.sum(
                qc_ds_event["QC_overall"] / len(qc_ds_event.time)
            )

            # Delta Z statistics over computable points
            dZ_mean[event] = np.mean(dz_r_nonan)
            dZ_med[event] = np.median(dz_r_nonan)
            dZ_q1[event] = np.quantile(dz_r_nonan, 0.25)
            dZ_q3[event] = np.quantile(dz_r_nonan, 0.75)
            dZ_min[event] = np.min(dz_r_nonan)
            dZ_max[event] = np.max(dz_r_nonan)

            event += 1
    # print(rain_accumulation, qc_ta_ratio, qc_ws_ratio, qc_wd_ratio, qc_vdsd_t_ratio)
    # print(dZ_mean, dZ_med, dZ_q1, dZ_q3, dZ_min, dZ_max)

    stats_ds["start_event"] = xr.DataArray(
        data=start_event, dims=["events"], attrs={"long_name": "event start epoch"}
    )
    stats_ds["end_event"] = xr.DataArray(
        data=end_event, dims=["events"], attrs={"long_name": "event end epoch"}
    )
    stats_ds["event_length"] = xr.DataArray(
        data=event_length,
        dims=["events"],
        attrs={"long_name": "event duration", "unit": "mn"},
    )
    stats_ds["rain_accumulation"] = xr.DataArray(
        data=rain_accumulation,
        dims=["events"],
        attrs={"long_name": "ams rain accumulation over the whole event", "unit": "mm"},
    )
    stats_ds["nb_dz_computable_pts"] = xr.DataArray(
        data=nb_dz_computable_pts,
        dims=["events"],
        attrs={
            "long_name": "number of timesteps for which Delta Z can be computed",
            "unit": "1",
        },
    )
    stats_ds["dZ_mean"] = xr.DataArray(
        data=dZ_mean,
        dims=["events"],
        attrs={
            "long_name": "average value of Delta Z for good timesteps",
            "unit": "dBZ",
        },
    )
    stats_ds["dZ_med"] = xr.DataArray(
        data=dZ_med,
        dims=["events"],
        attrs={
            "long_name": "median value of Delta Z for good timesteps",
            "unit": "dBZ",
        },
    )
    stats_ds["dZ_q1"] = xr.DataArray(
        data=dZ_q1,
        dims=["events"],
        attrs={
            "long_name": "first quartile of Delta Z distribution for good timesteps",
            "unit": "dBZ",
        },
    )
    stats_ds["dZ_q3"] = xr.DataArray(
        data=dZ_q3,
        dims=["events"],
        attrs={
            "long_name": "third quartile of Delta Z distribution for good timesteps",
            "unit": "dBZ",
        },
    )
    stats_ds["dZ_min"] = xr.DataArray(
        data=dZ_min,
        dims=["events"],
        attrs={
            "long_name": "minimum value of Delta Z for good timesteps",
            "unit": "dBZ",
        },
    )
    stats_ds["dZ_max"] = xr.DataArray(
        data=dZ_max,
        dims=["events"],
        attrs={
            "long_name": "maximum value of Delta Z for good timesteps",
            "unit": "dBZ",
        },
    )
    stats_ds["qc_ta_ratio"] = xr.DataArray(
        data=qc_ta_ratio,
        dims=["events"],
        attrs={
            "long_name": "ratio of timesteps where air temperature QC is good",
            "unit": "1",
            "comment": "among the timesteps for which Delta Z can be computed",
        },
    )
    stats_ds["qc_ws_ratio"] = xr.DataArray(
        data=qc_ws_ratio,
        dims=["events"],
        attrs={
            "long_name": "ratio of timesteps where wind speed QC is good",
            "unit": "1",
            "comment": "among the timesteps for which Delta Z can be computed",
        },
    )
    stats_ds["qc_wd_ratio"] = xr.DataArray(
        data=qc_wd_ratio,
        dims=["events"],
        attrs={
            "long_name": "ratio of timesteps where wind direction QC is good",
            "unit": "1",
            "comment": "among the timesteps for which Delta Z can be computed",
        },
    )
    stats_ds["qc_vdsd_t_ratio"] = xr.DataArray(
        data=qc_vdsd_t_ratio,
        dims=["events"],
        attrs={
            "long_name": "ratio of timesteps where check on relationship betwee, droplet fall speed and diameter is good",  # noqa E501
            "unit": "1",
            "comment": "among the timesteps for which Delta Z can be computed",
        },
    )
    stats_ds["qc_overall_ratio"] = xr.DataArray(
        data=qc_overall_ratio,
        dims=["events"],
        attrs={
            "long_name": "ratio of timesteps where all checks are good",  # noqa E501
            "unit": "1",
            "comment": "Checks combined : ta, ws, wd, vdsd_t, pr",
        },
    )

    return stats_ds


if __name__ == "__main__":
    pass
