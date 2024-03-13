"""Apply the processing from daily preprocessed files.

Input : Daily preprocessed files at days D and D-1
Output : Daily processed file for day D
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml
import xarray as xr

lgr = logging.getLogger(__name__)


def merge_preprocessed_data(yesterday, today, tomorrow):
    lgr.info("Beginning rain event selection")
    yesterday = xr.open_dataset(yesterday)
    today = xr.open_dataset(today)
    tomorrow = xr.open_dataset(tomorrow)
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
        if t[i + 1] - t[i] >= np.timedelta64(max_interval, "m"):
            if t[i] - start_candidate >= np.timedelta64(min_duration, "m"):
                start.append(start_candidate.values)
                end.append(t[i].values)
            start_candidate = t[i + 1]
    return start, end


def rain_event_selection_noweather(
    ds, conf
):  # with no constraint on cum for the moment
    sel_ds = ds.isel({"time": np.where(ds.disdro_pr.values > 0)[0]})

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
    if bool(ds["weather_data_avail"].values[0]) is True:
        start, end = rain_event_selection_weather(ds, conf)
    else:
        start, end = rain_event_selection_noweather(ds, conf)
    return start, end


def extract_dcr_data(ds, conf):
    # Extract DCR Ze at 3/4 relevant gates, extract DD Ze, compute Delta Ze
    # Get Doppler velocity at relevant gates, compute avg disdrometer fall speed(t)
    Ze_ds = xr.Dataset(coords=dict(time=(["time"], ds.time.data)))

    # DCR data extract
    ranges_to_keep = conf["plot_parameters"]["DCR_PLOTTED_RANGES"]
    Ze_ds["Zdcr"] = ds["Zdcr"].sel({"range": ranges_to_keep}, method="nearest")
    Ze_ds["DVdcr"] = ds["DVdcr"].sel({"range": ranges_to_keep}, method="nearest")
    # Disdrometer data extract
    Ze_ds["Zdd"] = ds["Zdlog_vfov_modv_tm"].sel(
        radar_frequencies=ds.radar_frequency, method="nearest"
    )
    Ze_ds["fallspeed_dd"] = xr.DataArray(
        data=np.nansum(
            np.nansum(ds["psd"].values, axis=2) * ds["measV"].values, axis=1
        ),
        dims=["time"],
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
        np.zeros(len(ds.time)), dims="time"
    )
    qc_ds["disdro_cum_since_event_begin"] = xr.DataArray(
        np.zeros(len(ds.time)), dims="time"
    )
    for s, e in zip(start, end):
        # mask = (qc_ds.time >= s) & (qc_ds.time <= e)
        # qc_ds["ams_cum_since_event_begin"] = qc_ds["ams_cum_since_event_begin"].where(
        #     ~mask, 1
        # )
        qc_ds["flag_event"].loc[slice(s, e)] = True

        qc_ds["ams_cum_since_event_begin"].loc[slice(s, e)] = (
            1 / 60 * np.nancumsum(ds["ams_pr"].sel(time=slice(s, e)).values)
        )
        qc_ds["disdro_cum_since_event_begin"].loc[slice(s, e)] = (
            1 / 60 * np.nancumsum(ds["disdro_pr"].sel(time=slice(s, e)).values)
        )

    # Flag for condition (rainfall_amount > N mm)
    qc_ds["QF_rainfall_amount"] = xr.DataArray(
        qc_ds["ams_cum_since_event_begin"] >= conf["thresholds"]["MIN_RAINFALL_AMOUNT"],
        dims="time",
    )

    # Temperature QC
    qc_ds["QC_ta"] = xr.DataArray(
        ds["ta"].values > conf["thresholds"]["MIN_TEMP"], dims="time"
    )
    # Wind speed and direction QCs
    qc_ds["QC_ws"] = xr.DataArray(
        ds["ws"].values < conf["thresholds"]["MAX_WS"], dims="time"
    )
    main_wind_dir = (conf["instrument_parameters"]["DD_ORIENTATION"] + 90) % 360
    dd_angle = conf["thresholds"]["DD_ANGLE"]
    x = 210
    print("HELLO")
    print(np.abs(x - main_wind_dir) < dd_angle)
    print(np.abs(x - main_wind_dir) > 360 - dd_angle)
    print(np.abs(x - (main_wind_dir + 180) % 360) < dd_angle)
    print(np.abs(x - (main_wind_dir + 180) % 360) > 360 - dd_angle)
    qc_ds["QC_wd"] = xr.DataArray(
        (np.abs(ds["wd"] - main_wind_dir) < dd_angle)
        | (np.abs(ds["wd"] - main_wind_dir) > 360 - dd_angle)
        | (np.abs(ds["wd"] - (main_wind_dir + 180) % 360) < dd_angle)
        | (np.abs(ds["wd"] - (main_wind_dir + 180) % 360) > 360 - dd_angle),
        dims="time",
    )  # data is between 0 and 360°

    # QC on AMS precipitation rate
    qc_ds["QC_pr"] = xr.DataArray(
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
            qc_ds["QC_pr"].loc[
                slice(start_time_chunk, stop_time_chunk - np.timedelta64(1, "m"))
            ] = RR_chunk <= conf["thresholds"]["MAX_RR"]

    # Check agreement between rain gauge and disdrometer rain measurements
    # extract ds(start to end), compute relative deviation and compare to conf tolerance
    qc_ds["QF_rg_dd"] = xr.DataArray(
        data=np.full(len(ds.time), False, dtype=bool), dims="time"
    )
    for s, e in zip(start, end):
        # event_mask = np.where((ds.time.values >= s) | (ds.time.values <= e))[0]
        qc_ds["QF_rg_dd"].loc[slice(s, e)] = (
            np.abs(
                qc_ds["disdro_cum_since_event_begin"].loc[slice(s, e)]
                - qc_ds["ams_cum_since_event_begin"].loc[slice(s, e)]
            )
            / qc_ds["ams_cum_since_event_begin"].loc[slice(s, e)]
            < conf["thresholds"]["DD_RG_MAX_PR_ACC_RATIO"]
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
        data=(np.abs(ratio_vdisdro_vth - 1) <= 0.3), dims="time"
    )

    # Overall QC : ta, ws, wd, ams_pr, v(d)
    qc_ds["QC_overall"] = (
        qc_ds["QC_ta"]
        & qc_ds["QC_ws"]
        & qc_ds["QC_wd"]
        & qc_ds["QC_pr"]
        & qc_ds["QC_vdsd_t"]
    )

    return qc_ds


def compute_quality_checks_noweather(ds, conf):
    qc_ds = xr.Dataset(coords=dict(time=(["time"], ds.time.data)))

    # flag the timesteps belonging to an event
    qc_ds["flag_event"] = xr.DataArray(
        data=np.full(len(ds.time), False, dtype=bool), dims=["time"]
    )
    # do a column for rain accumulation since last beginning of an event
    qc_ds["disdro_cum_since_event_begin"] = xr.DataArray(
        np.zeros(len(ds.time)), dims="time"
    )
    for s, e in zip(start, end):
        qc_ds["flag_event"].loc[slice(s, e)] = True
        qc_ds["disdro_cum_since_event_begin"].loc[slice(s, e)] = (
            1 / 60 * np.nancumsum(ds["disdro_pr"].sel(time=slice(s, e)).values)
        )

    # Flag for condition (rainfall_amount > N mm)
    qc_ds["QF_rainfall_amount"] = xr.DataArray(
        qc_ds["disdro_cum_since_event_begin"]
        >= conf["thresholds"]["MIN_RAINFALL_AMOUNT"],
        dims="time",
    )

    # QC on DISDROMETER precipitation rate
    qc_ds["QC_pr"] = xr.DataArray(
        data=ds["disdro_pr"] < conf["thresholds"]["MAX_RR"], dims=["time"]
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
        data=(np.abs(ratio_vdisdro_vth - 1) <= 0.3), dims="time"
    )

    return qc_ds


def compute_quality_checks(ds, conf, start, end):
    if bool(ds["weather_data_avail"].values[0]) is True:
        qc_ds = compute_quality_checks_weather(ds, conf, start, end)
        lgr.info("Compute QC dataset (case with weather)")
    else:
        qc_ds = compute_quality_checks_noweather(ds, conf, start, end)
        lgr.info("Compute QC dataset (case without weather)")
    return qc_ds


def compute_todays_events_stats_weather(Ze_ds, conf, qc_ds, start, end):
    dicos = []
    for s, e in zip(start, end):
        if (
            pd.to_datetime(s).day
            == pd.to_datetime(ds.time.isel(time=len(qc_ds.time) // 2).values).day
        ):
            dico = {}
            r = conf["instrument_parameters"]["DCR_DZ_RANGE"]
            dz_r = Ze_ds["Delta_Z"].sel(time=slice(s, e)).sel(range=r, method="nearest")
            dz_r_nonan = dz_r[np.isfinite(dz_r)]
            # General info about the event
            event_length = (e - s) / np.timedelta64(1, "m") + 1
            rain_accumulation = qc_ds["ams_cum_since_event_begin"].loc[e]
            nb_dz_computable_pts = len(dz_r)
            # QC passed ratios
            qc_ds_event = qc_ds.sel(time=slice(s, e)).loc[{"time": np.isfinite(dz_r)}]
            print(qc_ds_event.QC_ta.values.shape)
            print(
                "event length : ",
                event_length,
                "finite points : ",
                nb_dz_computable_pts,
            )
            qc_ta_ratio = np.sum(qc_ds_event["QC_ta"]) / len(qc_ds_event.time)
            qc_ws_ratio = np.sum(qc_ds_event["QC_ws"]) / len(qc_ds_event.time)
            qc_wd_ratio = np.sum(qc_ds_event["QC_wd"]) / len(qc_ds_event.time)
            qc_vdsd_t_ratio = np.sum(qc_ds_event["QC_vdsd_t"]) / len(qc_ds_event.time)
            # print(qc_ta_ratio, qc_ws_ratio, qc_wd_ratio, qc_vdsd_t_ratio)
            # Delta Z statistics over computable points
            dZ_mean = np.mean(dz_r_nonan)
            dZ_med = np.median(dz_r_nonan)
            dZ_q1 = np.quantile(dz_r_nonan, 0.25)
            dZ_q3 = np.quantile(dz_r_nonan, 0.75)
            dZ_min = np.min(dz_r_nonan)
            dZ_max = np.max(dz_r_nonan)

            dicos.append(dico)
    print(rain_accumulation, qc_ta_ratio, qc_ws_ratio, qc_wd_ratio, qc_vdsd_t_ratio)
    print(dZ_mean, dZ_med, dZ_q1, dZ_q3, dZ_min, dZ_max)
    # return event_stats_ds
    return dicos


def store_outputs(ds, conf):
    pass


if __name__ == "__main__":
    yesterday = (
        "../tests/data/outputs/palaiseau_2022-10-13_basta-parsivel-ws_preprocessed.nc"
    )
    today = (
        "../tests/data/outputs/palaiseau_2022-10-14_basta-parsivel-ws_preprocessed.nc"
    )
    tomorrow = (
        "../tests/data/outputs/palaiseau_2022-10-15_basta-parsivel-ws_preprocessed.nc"
    )
    conf = toml.load("../tests/data/conf/config_palaiseau_basta-parsivel-ws.toml")

    y = xr.open_dataset(yesterday)
    # print(y.dims)

    ds = merge_preprocessed_data(yesterday, today, tomorrow)
    start, end = rain_event_selection(ds, conf)

    Ze_ds = extract_dcr_data(ds, conf)
    # print(Ze_ds)
    qc_ds = compute_quality_checks(ds, conf, start, end)
    events_stats_ds = compute_todays_events_stats_weather(
        Ze_ds, conf, qc_ds, start, end
    )

    plt.figure()
    plt.plot(
        qc_ds.time,
        qc_ds.ams_cum_since_event_begin.values,
        color="blue",
        label="ams rainfall amount",
    )
    plt.plot(
        qc_ds.time,
        qc_ds.disdro_cum_since_event_begin.values,
        color="red",
        label="disdro rainfall amount",
    )
    plt.legend()
    plt.savefig("./plot_diagnostic_preprocessing.png", dpi=300)
    plt.close()

    plt.figure()
    # plt.plot(qc_ds.time, qc_ds.QC_ta, label="ta", alpha=0.4)
    # plt.plot(qc_ds.time, qc_ds.QC_ws, label="ws", alpha=0.4)
    plt.plot(qc_ds.time, 225 + 10 * qc_ds.QC_wd, label="qc_wd", alpha=1)
    # plt.plot(qc_ds.time, qc_ds.QC_pr, label="pr", alpha=0.4)
    # plt.plot(qc_ds.time, qc_ds.QC_vdsd_t, label="vd", alpha=0.4)
    # plt.plot(qc_ds.time, qc_ds.QC_overall, label="overall")
    # plt.plot(ds.time, ds.ws)
    plt.plot(ds.time, ds.wd)
    plt.axhline(y=225, alpha=0.3)
    plt.xlim(left=start[0], right=end[0])
    plt.legend()
    plt.savefig("./plot_diagnostic_preprocessing2.png", dpi=300)
    plt.close()
