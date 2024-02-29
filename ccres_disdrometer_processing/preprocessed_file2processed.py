"""Apply the processing from daily preprocessed files.

Input : Daily preprocessed files at days D and D-1
Output : Daily processed file for day D
"""
import logging

import numpy as np
import xarray as xr

lgr = logging.getLogger(__name__)


def merge_preprocessed_data(yesterday, today, tomorrow):
    lgr.info("Beginning rain event selection")
    ds = xr.concat((yesterday, today, tomorrow), dim="time")
    return ds


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
        if t[i + 1] - t[i] > np.timedelta64(max_interval, "m"):
            if t[i] - start_candidate >= np.timedelta64(min_duration, "m"):
                start.append(start_candidate.values)
                end.append(t[i].values)
            start_candidate = t[i + 1]
    return start, end


def rain_event_selection_noweather(
    ds, conf
):  # with no constraint on cum for the moment
    sel_ds = ds.isel({"time": np.where(ds.dd_pr.values > 0)[0]})

    min_duration, max_interval = (
        conf["thresholds"]["MIN_DURATION"],
        conf["thresholds"]["MAX_INTERVAL"],
    )

    t = sel_ds.time
    start, end = [], []
    start_candidate = t[0]
    for i in range(len(t) - 1):
        if t[i + 1] - t[i] > np.timedelta64(max_interval, "m"):
            if t[i] - start_candidate >= np.timedelta64(min_duration, "m"):
                start.append(start_candidate.values)
                end.append(t[i].values)
            start_candidate = t[i + 1]
    return start, end


def rain_event_selection(ds, conf):
    if bool(ds.attrs["weather_avail"]) is True:
        rain_event_selection_weather(ds)
    else:
        rain_event_selection_noweather(ds)
    return True


def extract_dcr_data(ds, conf):
    # Extract DCR Ze at 3/4 relevant gates, extract DD Ze, compute Delta Ze
    # Get Doppler velocity at relevant gates, compute avg disdrometer fall speed(t)

    # DCR data extract
    ranges_to_keep = conf["plot_parameters"]["DCR_PLOTTED_RANGES"]
    Zdcr = ds["Zdcr"].sel({"range": ranges_to_keep}, method="nearest")
    DVdcr = ds["DVdcr"].sel({"range": ranges_to_keep}, method="nearest")
    # Disdrometer data extract
    Zdd = ds["Zdlog_vfov_modv_tm"]
    fallspeed_dd = np.nansum(
        np.nansum(ds["psd"].values, axis=2) * ds["measV"].values, axis=1
    )
    # Delta Ze
    Delta_Z = Zdcr - Zdd

    return Zdcr, DVdcr, Zdd, fallspeed_dd, Delta_Z


def compute_quality_checks_weather(ds, conf, start, end):
    # flag the timesteps belonging to an event
    flag_event = 0
    # do a column for rain accumulation since last beginning of an event
    flag_rain_acc = 0
    # Weather quality flags
    QF_ta = ds["ta"].values > conf["thresholds"]["MIN_TEMP"]
    QF_ws = ds["ws"].values > conf["thresholds"]["MAX_WS"]
    main_wind_dir = (conf["instrument_parameters"]["DD_ORIENTATION"] + 90) % 360
    dd_angle = conf["thresholds"]["DD_ANGLE"]
    QF_wd = (np.abs(ds["wd"] - main_wind_dir) < dd_angle) | (
        np.abs(ds["wd"] - (main_wind_dir + 180) % 360) < dd_angle
    )
    QF_ams_pr = ds["ams_pr"].values < conf["thresholds"]["MAX_RR"]

    QF_rg_dd = np.zeros(len(ds.time))
    for s, e in zip(start, end):
        # extract ds(start to end), compute relative deviation
        # and compare to the tolerance specified in the conf (?)
        break

    pass


def compute_quality_checks_noweather(ds, conf):
    pass


def compute_todays_events_stats(ds, conf):
    pass


def store_outputs(ds, conf):
    pass
