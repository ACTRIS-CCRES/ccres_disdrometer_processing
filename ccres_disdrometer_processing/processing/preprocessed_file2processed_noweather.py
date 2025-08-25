"""Apply the processing from daily preprocessed files.

Input : Daily preprocessed files at days D and D-1
Output : Daily processed file for day D

"""

import logging

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import linregress

lgr = logging.getLogger(__name__)

QC_FILL_VALUE = -9


def rmse(y, y_hat):  # rmse func to remove dependency from sklearn rmse
    return np.sqrt(((y - y_hat) ** 2).mean())


def rain_event_selection_noweather(ds, conf):
    sel_ds = ds.isel({"time": np.where(ds.disdro_pr.values > 0)[0]})

    min_duration, max_interval, min_rainfall_amount = (
        conf["thresholds"]["MIN_DURATION"],
        conf["thresholds"]["MAX_INTERVAL"],
        conf["thresholds"]["MIN_RAINFALL_AMOUNT"],
    )
    t = sel_ds.time
    start, end = [], []
    if t.size <= 1:
        return start, end
    start_candidate = t[0]
    for i in range(t.size - 1):
        if i + 1 == t.size - 1:
            if t[i + 1] - t[i] > np.timedelta64(max_interval, "m"):
                if t[i] - start_candidate >= np.timedelta64(min_duration, "m"):
                    start.append(start_candidate.values)
                    end.append(t[i].values)
            elif t[i + 1] - start_candidate >= np.timedelta64(min_duration, "m"):
                start.append(start_candidate.values)
                end.append(t[i + 1].values)
        elif t[i + 1] - t[i] > np.timedelta64(max_interval, "m"):
            if t[i] - start_candidate >= np.timedelta64(min_duration, "m"):
                start.append(start_candidate.values)
                end.append(t[i].values)
            start_candidate = t[i + 1]
    for s, e in zip(start.copy(), end.copy()):
        rain_accumulation_event = (
            1 / 60 * np.nansum(sel_ds.disdro_pr.sel(time=slice(s, e)))
        )
        if rain_accumulation_event < min_rainfall_amount:
            start.remove(s)
            end.remove(e)
    return start, end


def compute_quality_checks_noweather(ds, conf, start, end):
    qc_ds = xr.Dataset(coords=dict(time=(["time"], ds.time.data)))

    # flag the timesteps belonging to an event
    qc_ds["flag_event"] = xr.DataArray(
        data=np.full(ds.time.size, 0, dtype="i2"),
        dims=["time"],
        attrs={
            "long_name": "flag to describe if a timestep belongs to a rain event",
        },
    )
    # do a column for rain accumulation since last beginning of an event
    qc_ds["ams_cp_since_event_begin"] = xr.DataArray(
        np.nan * np.zeros(ds.time.size).astype(np.single),
        dims=["time"],
        attrs={
            "long_name": "pluviometer rain accumulation since last begin of an event",
            "units": "mm",
            "comment": "not computable when no AMS data is provided",
        },
    )
    qc_ds["disdro_cp_since_event_begin"] = xr.DataArray(
        np.zeros(ds.time.size).astype(np.single),
        dims="time",
        attrs={
            "long_name": "disdrometer rain accumulation since last begin of an event",
            "units": "mm",
        },
    )
    for s, e in zip(start, end):
        qc_ds["flag_event"].loc[slice(s, e)] = 1
        qc_ds["disdro_cp_since_event_begin"].loc[slice(s, e)] = (
            1 / 60 * np.nancumsum(ds["disdro_pr"].sel(time=slice(s, e)).values)
        )

    # Flag for condition (rainfall_amount > N mm)
    qc_ds["QF_rainfall_amount"] = xr.DataArray(
        (
            qc_ds["disdro_cp_since_event_begin"]
            >= conf["thresholds"]["MIN_RAINFALL_AMOUNT"]
        ).astype("i2"),
        dims="time",
        attrs={
            "long_name": "Quality flag for minimum rainfall amount",
            "comment": f"Flag based on disdrometer data ; threshold = {conf['thresholds']['MIN_RAINFALL_AMOUNT']:.0f} mm",  # noqa E501
        },
    )

    # QC on DISDROMETER precipitation rate
    qc_ds["QC_pr"] = xr.DataArray(
        data=(ds["disdro_pr"] < conf["thresholds"]["MAX_RR"]).astype("i2"),
        dims=["time"],
        attrs={
            "long_name": "Quality check for rainfall rate (based on disdro pr data)",
            "comment": f"threshold = {conf['thresholds']['MAX_RR']:.0f} mm/h",
        },
    )

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
        data=(np.abs(ratio_vdisdro_vth - 1) <= 0.3).astype("i2"),
        dims="time",
        attrs={
            "long_name": "Quality check for cohence between fall speed and droplet diameter",  # noqa E501
            "comment": f"threshold = {conf['thresholds']['DD_FALLSPEED_RATIO']:.1f} i.e. {100 * conf['thresholds']['DD_FALLSPEED_RATIO']:.0f}% of relative error between average fall speed computed from speed distribution and average fall speed modeled from diameter distribution",  # noqa E501,
        },
    )

    # Create non-computable QC variables in the dataset
    for key in ["QC_ta", "QC_wd", "QC_ws", "QC_hur", "QF_rg_dd"]:
        qc_ds[key] = xr.DataArray(
            data=QC_FILL_VALUE * np.ones(ds.time.size).astype("i2"),
            dims="time",
            attrs={
                "comment": "not computable when no AMS data is provided",
                "flag_values": np.array([0, 1]).astype("i2"),
            },
        )
    qc_ds["QC_ta"].attrs["long_name"] = "Quality check for air temperature"
    qc_ds["QC_wd"].attrs["long_name"] = "Quality check for wind direction"
    qc_ds["QC_ws"].attrs["long_name"] = "Quality check for wind speed"
    qc_ds["QC_hur"].attrs["long_name"] = "Quality check for relative humidity"
    qc_ds["QF_rg_dd"].attrs["long_name"] = (
        "Quality flag for discrepancy between rain gauge and disdrometer precipitation rate"  # noqa
    )
    qc_ds["QC_ta"].attrs["flag_meanings"] = "temperature_under_threshold temperature_ok"
    qc_ds["QC_wd"].attrs["flag_meanings"] = (
        "wind_direction_out_of_good_range wind_direction_within_good_range"
    )
    qc_ds["QC_ws"].attrs["flag_meanings"] = "wind_speed_over_threshold wind_speed_ok"
    qc_ds["QC_hur"].attrs["flag_meanings"] = (
        "hur_above_lower_bound hur_over_upper_bound"
    )
    qc_ds["QF_rg_dd"].attrs["flag_meanings"] = (
        "discrepancy_rain_gauge_disdrometer_above_threshold discrepancy_rain_gauge_disdrometer_ok"  # noqa
    )

    # Overall QC : disdro_pr, v(d)
    qc_ds["QC_overall"] = xr.DataArray(
        data=qc_ds["QC_pr"] & qc_ds["QC_vdsd_t"],
        dims="time",
        attrs={
            "long_name": "Overall quality check",
            "comment": "Checks combined : vdsd_t, pr",
        },
    )

    # Data attributes and types
    for key in [
        "flag_event",
        "QF_rainfall_amount",
        "QC_pr",
        "QC_vdsd_t",
        "QC_overall",
    ]:
        qc_ds[key].attrs["flag_values"] = np.array([0, 1]).astype("i2")

    qc_ds["flag_event"].attrs["flag_meanings"] = (
        "timestep_not_involved_in_any_event timestep_part_of_an_event"
    )
    qc_ds["QF_rainfall_amount"].attrs["flag_meanings"] = (
        "less_rain_than_threshold_since_event_begin more_rain_than_threshold_since_event_begin"  # noqa E501
    )
    qc_ds["QC_pr"].attrs["flag_meanings"] = (
        "precipitation_rate_above threshold precipitation_rate_ok"
    )
    qc_ds["QC_vdsd_t"].attrs["flag_meanings"] = (
        "discrepancy_between_observed_and_modeled_disdrometer_droplet_fallspeed_above_threshold discrepancy_under_threshold"  # noqa e501
    )
    qc_ds["QC_overall"].attrs["flag_meanings"] = "at_least_one_QC_not_OK all_QC_OK"

    return qc_ds


def compute_todays_events_stats_noweather(
    ds, Ze_ds, conf, qc_ds, start, end, day_today
):
    n = 0
    for e in end:
        if pd.to_datetime(e).day == day_today:
            n += 1
    # n is the number of events to store in the dataset
    # i.e. the number of events which end at day D

    stats_ds = xr.Dataset(
        coords=dict(events=(["events"], np.linspace(1, n, n, dtype="int32")))
    )

    dZ_mean, dZ_med, dZ_q1, dZ_q3, dZ_min, dZ_max = (
        np.zeros(n, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
    )
    event_length, rain_accumulation, qf_rain, nb_dz_computable_pts, nb_good_points = (
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
    )
    qc_vdsd_t_ratio, qc_pr_ratio, qc_overall_ratio = (
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
    )
    start_event, end_event = (
        np.empty(n, dtype="datetime64[ns]"),
        np.empty(n, dtype="datetime64[ns]"),
    )

    slope, intercept, r2, rms_error = (
        np.zeros(n, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
    )

    event = 0
    for s, e in zip(start, end):
        if pd.to_datetime(e).day == day_today:
            start_event[event] = s
            end_event[event] = e
            r = conf["instrument_parameters"]["DCR_DZ_RANGE"]
            dz_r = Ze_ds["Delta_Z"].sel(time=slice(s, e)).sel(range=r, method="nearest")
            dz_r_nonan = dz_r[np.isfinite(dz_r)]
            print("event length :", (e - s) / np.timedelta64(1, "m") + 1)
            print("Finite points : ", len(np.where(np.isfinite(dz_r))[0]))
            # General info about the event
            event_length[event] = (e - s) / np.timedelta64(1, "m") + 1
            rain_accumulation[event] = qc_ds["disdro_cp_since_event_begin"].loc[e]
            qf_rain[event] = qc_ds["QF_rainfall_amount"].loc[e]
            nb_dz_computable_pts[event] = len(dz_r_nonan)
            # Qc passed ratios
            qc_ds_event = qc_ds.sel(time=slice(s, e)).loc[{"time": np.isfinite(dz_r)}]
            qc_pr_ratio[event] = np.sum(qc_ds_event["QC_pr"]) / qc_ds_event.time.size
            qc_vdsd_t_ratio[event] = (
                np.sum(qc_ds_event["QC_vdsd_t"]) / qc_ds_event.time.size
            )
            qc_overall_ratio[event] = np.sum(
                qc_ds_event["QC_overall"] / qc_ds_event.time.size
            )
            good_points_event = np.sum(qc_ds_event["QC_overall"])
            nb_good_points[event] = good_points_event

            dz_r_good = dz_r_nonan.isel(
                time=np.where(qc_ds_event["QC_overall"] == 1)[0]
            )
            if good_points_event > 0:
                # Delta Z statistics over computable points
                dZ_mean[event] = np.mean(dz_r_good)
                dZ_med[event] = np.median(dz_r_good)
                dZ_q1[event] = np.quantile(dz_r_good, 0.25)
                dZ_q3[event] = np.quantile(dz_r_good, 0.75)
                dZ_min[event] = np.min(dz_r_good)
                dZ_max[event] = np.max(dz_r_good)

                # Stats for regression Zdcr/Zdd
                z_dcr_nonan = (
                    Ze_ds["Zdcr"]
                    .sel(time=slice(s, e))
                    .sel(range=r, method="nearest")[np.isfinite(dz_r)]
                )
                z_dd_nonan = Ze_ds["Zdd"].sel(time=slice(s, e))[np.isfinite(dz_r)]
                z_dcr_nonan = z_dcr_nonan.isel(
                    time=np.where(qc_ds_event["QC_overall"] == 1)[0]
                ).values.reshape(
                    (-1, 1)
                )  # keep only QC passed timesteps for regression
                z_dd_nonan = z_dd_nonan.isel(
                    time=np.where(qc_ds_event["QC_overall"] == 1)[0]
                ).values.reshape(
                    (-1, 1)
                )  # keep only QC passed timesteps for regression

                slope_event, intercept_event, r_event, p, se = linregress(
                    z_dd_nonan.flatten(), z_dcr_nonan.flatten()
                )
                z_dcr_hat = intercept_event + slope_event * z_dd_nonan
                slope[event] = slope_event
                intercept[event] = intercept_event
                r2[event] = r_event**2
                rms_error[event] = rmse(z_dcr_nonan, z_dcr_hat)

            event += 1
    stats_ds["start_event"] = xr.DataArray(
        data=start_event, dims=["events"], attrs={"long_name": "event start date"}
    )
    stats_ds["end_event"] = xr.DataArray(
        data=end_event, dims=["events"], attrs={"long_name": "event end date"}
    )
    stats_ds["event_length"] = xr.DataArray(
        data=event_length.astype("i4"),
        dims=["events"],
        attrs={"long_name": "event duration", "units": "mn"},
    )
    stats_ds["rain_accumulation"] = xr.DataArray(
        data=rain_accumulation.astype(np.float32),
        dims=["events"],
        attrs={
            "long_name": "disdrometer rain accumulation over the whole event",
            "units": "mm",
        },
    )
    stats_ds["QF_rain_accumulation"] = xr.DataArray(
        data=qf_rain.astype("i2"),
        dims=["events"],
        attrs={
            "long_name": "Flag on event rain accumulation",
            "flag_values": np.array([0, 1]).astype("i2"),
            "flag_meanings": "accumulation_above_threshold accumulation_over_threshold",
        },
    )

    stats_ds["QF_rg_dd_event"] = xr.DataArray(
        data=QC_FILL_VALUE * np.ones(len(stats_ds.events)).astype("i2"),
        dims=["events"],
        attrs={
            "long_name": "Flag on deviation between rain gauge and disdrometer precipitation rate at the end of an event",  # noqa
            "flag_values": np.array([0, 1]).astype("i2"),
            "flag_meanings": "accumulation_above_threshold accumulation_over_threshold",  # noqa
            "comment": "not computable when no AMS data is provided",
        },
    )
    stats_ds["nb_dz_computable_pts"] = xr.DataArray(
        data=nb_dz_computable_pts.astype("i4"),
        dims=["events"],
        attrs={
            "long_name": "number of timesteps for which Delta Z can be computed",
        },
    )

    stats_ds["QC_vdsd_t_ratio"] = xr.DataArray(
        data=qc_vdsd_t_ratio.astype(np.float32),
        dims=["events"],
        attrs={
            "long_name": "ratio of timesteps where check on relationship between droplet fall speed and diameter is good",  # noqa E501
            "comment": "among the timesteps for which Delta Z can be computed",
        },
    )
    stats_ds["QC_pr_ratio"] = xr.DataArray(
        data=qc_pr_ratio.astype(np.float32),
        dims=["events"],
        attrs={
            "long_name": "ratio of timesteps where precipitation rate QC is good",  # noqa E501
            "comment": "among the timesteps for which Delta Z can be computed",
        },
    )

    for key in ["QC_ta_ratio", "QC_ws_ratio", "QC_wd_ratio"]:
        stats_ds[key] = xr.DataArray(
            data=np.nan * np.zeros(stats_ds.events.size).astype(np.float32),
            dims=["events"],
            attrs={
                "comment": "not computable when no AMS data is provided",
            },
        )
    stats_ds["QC_ta_ratio"].attrs["long_name"] = (
        "ratio of timesteps where check for air temperature is good"
    )
    stats_ds["QC_wd_ratio"].attrs["long_name"] = (
        "ratio of timesteps where check for wind direction is good"
    )
    stats_ds["QC_ws_ratio"].attrs["long_name"] = (
        "ratio of timesteps where check for wind speed is good"
    )

    stats_ds["QC_overall_ratio"] = xr.DataArray(
        data=qc_overall_ratio.astype(np.float32),
        dims=["events"],
        attrs={
            "long_name": "ratio of timesteps where all checks are good",  # noqa E501
            "comment": "Checks combined : vdsd_t, pr",
        },
    )
    stats_ds["good_points_number"] = xr.DataArray(
        data=nb_good_points.astype("i4"),
        dims=["events"],
        attrs={
            "long_name": "number of timesteps where all checks are good",
        },
    )

    stats_ds["dZ_mean"] = xr.DataArray(
        data=dZ_mean,
        dims=["events"],
        attrs={
            "long_name": "average value of Delta Z for good timesteps",
            "units": "dBZ",
        },
    )
    stats_ds["dZ_med"] = xr.DataArray(
        data=dZ_med,
        dims=["events"],
        attrs={
            "long_name": "median value of Delta Z for good timesteps",
            "units": "dBZ",
        },
    )
    stats_ds["dZ_q1"] = xr.DataArray(
        data=dZ_q1,
        dims=["events"],
        attrs={
            "long_name": "first quartile of Delta Z distribution for good timesteps",
            "units": "dBZ",
        },
    )
    stats_ds["dZ_q3"] = xr.DataArray(
        data=dZ_q3,
        dims=["events"],
        attrs={
            "long_name": "third quartile of Delta Z distribution for good timesteps",
            "units": "dBZ",
        },
    )
    stats_ds["dZ_min"] = xr.DataArray(
        data=dZ_min,
        dims=["events"],
        attrs={
            "long_name": "minimum value of Delta Z for good timesteps",
            "units": "dBZ",
        },
    )
    stats_ds["dZ_max"] = xr.DataArray(
        data=dZ_max,
        dims=["events"],
        attrs={
            "long_name": "maximum value of Delta Z for good timesteps",
            "units": "dBZ",
        },
    )

    stats_ds["reg_slope"] = xr.DataArray(
        data=slope,
        dims=["events"],
        attrs={
            "long_name": "slope of the linear regression Zdd/Zdcr for each event",
        },
    )
    stats_ds["reg_intercept"] = xr.DataArray(
        data=intercept,
        dims=["events"],
        attrs={
            "long_name": "intercept of the linear regression Zdd/Zdcr for each event",
            "comment": "expected to be related to the bias i.e. to mean Delta Z",
        },
    )

    stats_ds["reg_score"] = xr.DataArray(
        data=r2,
        dims=["events"],
        attrs={
            "long_name": "R_squared of the linear regression Zdd/Zdcr for each event",
        },
    )

    stats_ds["reg_rmse"] = xr.DataArray(
        data=rms_error,
        dims=["events"],
        attrs={
            "long_name": "RMSE of the linear regression Zdd/Zdcr for each event",
        },
    )

    return stats_ds


if __name__ == "__main__":
    pass
