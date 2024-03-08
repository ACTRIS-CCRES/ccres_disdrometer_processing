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
    Ze_ds = xr.Dataset(coords=dict(time=(["time"], ds.time.data)))

    # DCR data extract
    ranges_to_keep = conf["plot_parameters"]["DCR_PLOTTED_RANGES"]
    Ze_ds["Zdcr"] = ds["Zdcr"].sel({"range": ranges_to_keep}, method="nearest")
    Ze_ds["DVdcr"] = ds["DVdcr"].sel({"range": ranges_to_keep}, method="nearest")
    # Disdrometer data extract
    Ze_ds["Zdd"] = ds["Zdlog_vfov_modv_tm"]
    Ze_ds["fallspeed_dd"] = np.nansum(
        np.nansum(ds["psd"].values, axis=2) * ds["measV"].values, axis=1
    )
    # Delta Ze
    Ze_ds["Delta_Z"] = Ze_ds["Zdcr"] - Ze_ds["Zdd"]

    return Ze_ds


def compute_quality_checks_weather(ds, conf, start, end):
    qc_ds = xr.Dataset(coords=dict(time=(["time"], ds.time.data)))

    # flag the timesteps belonging to an event
    qc_ds["flag_event"] = xr.DataArray(
        data=np.full(len(ds.time), False, dtype=bool), dims=["time"]
    )
    # do a column for rain accumulation since last beginning of an event
    qc_ds["ams_cum_since_event_begin"] = xr.DataArray(
        np.zeros(len(ds.time)), dim="time"
    )
    qc_ds["dd_cum_since_event_begin"] = xr.DataArray(np.zeros(len(ds.time)), dim="time")
    for s, e in zip(start, end):
        qc_ds["flag_event"].sel(time=slice(s, e)).values = True
        qc_ds["ams_cum_since_event_begin"].sel(time=slice(s, e)).values = (
            1 / 60 * np.cumsum(ds["ams_pr"].sel(time=slice(s, e)).values)
        )
        qc_ds["dd_cum_since_event_begin"].sel(time=slice(s, e)).values = (
            1 / 60 * np.cumsum(ds["dd_pr"].sel(time=slice(s, e)).values)
        )

    # Flag for condition (rainfall_amount > N mm)
    qc_ds["QF_ams_rainfall_amount"] = xr.DataArray(
        ds["ams_cum_since_event_begin"] >= conf["thresholds"]["MIN_RAINFALL_AMOUNT"],
        dim="time",
    )

    # Temperature QC
    qc_ds["QC_ta"] = xr.DataArray(
        ds["ta"].values > conf["thresholds"]["MIN_TEMP"], dim="time"
    )
    # Wind speed and direction QCs
    qc_ds["QC_ws"] = xr.DataArray(
        ds["ws"].values > conf["thresholds"]["MAX_WS"], dim="time"
    )
    main_wind_dir = (conf["instrument_parameters"]["DD_ORIENTATION"] + 90) % 360
    dd_angle = conf["thresholds"]["DD_ANGLE"]
    qc_ds["QC_wd"] = xr.DataArray(
        (np.abs(ds["wd"] - main_wind_dir) < dd_angle)
        | (np.abs(ds["wd"] - (main_wind_dir + 180) % 360) < dd_angle),
        dim="time",
    )
    # QC on AMS precipitation rate
    qc_ds["QC_ams_pr"] = xr.DataArray(
        data=np.full(len(ds.time), True, dtype=bool), dims=["time"]
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

            print(
                qc_ds["QC_ams_pr"]
                .sel(
                    {
                        "time": slice(
                            start_time_chunk, stop_time_chunk - np.timedelta64(1, "m")
                        )
                    }
                )
                .values.shape
            )

            qc_ds["QC_ams_pr"].sel(
                {
                    "time": slice(
                        start_time_chunk, stop_time_chunk - np.timedelta64(1, "m")
                    )
                }
            ).values = np.tile(
                (RR_chunk <= conf["thresholds"]["MAX_RR"]),
                conf["thresholds"]["PR_SAMPLING"],
            )

    # Check agreement between rain gauge and disdrometer rain measurements
    # extract ds(start to end), compute relative deviation and compare to conf tolerance
    qc_ds["QF_rg_dd"] = xr.DataArray(np.zeros(len(ds.time)), dim="time")
    for s, e in zip(start, end):
        event_mask = np.where((ds.time.values >= s) | (ds.time.values <= e))[0]
        qc_ds["QF_rg_dd"].isel({"time": event_mask}).values = (
            np.abs(
                ds["dd_cum_since_event_begin"].isel({"time": event_mask}).values
                - ds["ams_cum_since_event_begin"].isel({"time": event_mask}).values
            )
            / ds["ams_cum_since_event_begin"].isel({"time": event_mask}).values
            < conf["thresholds"]["DD_RG_MAX_PR_ACC_RATIO"]
        )

    # QC relationship v(dsd)
    qc_ds["QC_vdsd_t"] = xr.DataArray(
        data=np.full(len(ds.time), False, dtype=bool), dims=["time"]
    )

    vth_disdro = (np.nansum(ds["psd"].values, axis=2) @ ds["modV"].values) / np.nansum(
        ds["psd"].values, axis=(1, 2)
    )  # average fall speed weighed by (num_drops_per_time_and_diameter)
    vobs_disdro = (
        np.nansum(ds["psd"].values, axis=1) @ ds["speed_classes"].values
    ) / np.nansum(ds["psd"].values, axis=(1, 2))
    ratio_vdisdro_vth = vobs_disdro / vth_disdro

    qc_ds["QC_vdsd_t"] = xr.DataArray(
        data=(np.abs(ratio_vdisdro_vth - 1) <= 0.3), dim="time"
    )

    return qc_ds


def compute_quality_checks_noweather(ds, conf):
    QF_dd_pr = ds["ams_pr"].values < conf["thresholds"]["MAX_RR"]
    return QF_dd_pr


def compute_todays_events_stats(ds, conf, start, end):
    event_stats_ds = xr.Dataset(coords=dict(time=(["time"], ds.time.data)))
    return event_stats_ds


def store_outputs(ds, conf):
    pass


if __name__ == "__main__":
    yesterday = "/homedata/ygrit/"
    today = 0
    tomorrow = 0
    conf = 0
    ds = merge_preprocessed_data(yesterday, today, tomorrow)
    x = rain_event_selection(ds, conf)
    print(x)
