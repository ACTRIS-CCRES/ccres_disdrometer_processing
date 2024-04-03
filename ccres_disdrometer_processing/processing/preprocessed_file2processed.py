"""Apply the processing from daily preprocessed files.

Input : Daily preprocessed files at days D and D-1
Output : Daily processed file for day D
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import preprocessed_file2processed_noweather as processing_noweather
import preprocessed_file2processed_weather as processing
import toml
import xarray as xr

TIME_UNITS = "seconds since 2000-01-01T00:00:00.0Z"
TIME_CALENDAR = "standard"

lgr = logging.getLogger(__name__)


def merge_preprocessed_data(yesterday, today, tomorrow):
    lgr.info("Beginning rain event selection")
    yesterday = xr.open_dataset(yesterday)
    today = xr.open_dataset(today)
    tomorrow = xr.open_dataset(tomorrow)
    ds = xr.concat((yesterday, today, tomorrow), dim="time")
    return ds


def rain_event_selection(ds, conf):
    if bool(ds["weather_data_avail"].values[0]) is True:
        start, end = processing.rain_event_selection_weather(ds, conf)
    else:
        start, end = processing_noweather.rain_event_selection_noweather(ds, conf)
    return start, end


def extract_dcr_data(ds, conf):
    # Extract DCR Ze at 3/4 relevant gates, extract DD Ze, compute Delta Ze
    # Get Doppler velocity at relevant gates, compute avg disdrometer fall speed(t)
    ranges_to_keep = conf["plot_parameters"]["DCR_PLOTTED_RANGES"]
    Ze_ds = xr.Dataset(
        coords=dict(
            time=(["time"], ds.time.data),
            range=(["range"], np.array(ranges_to_keep)),
        )
    )

    # DCR data extract
    Ze_ds["Zdcr"] = xr.DataArray(
        data=ds["Zdcr"].sel({"range": ranges_to_keep}, method="nearest").data,
        dims=["time", "range"],
        attrs={
            "long_name": "DCR reflectivity at the ranges closest to those defined in the station configuration file",  # noqa E501
            "units": "dBZ",
        },
    )
    Ze_ds["DVdcr"] = xr.DataArray(
        data=ds["DVdcr"].sel({"range": ranges_to_keep}, method="nearest").data,
        dims=["time", "range"],
        attrs={
            "long_name": "DCR Doppler velocity",
            "units": "m.s^-1",
            "comment": "available at the ranges closest to those defined in the station configuration file",  # noqa E501
        },
    )
    # Disdrometer data extract
    Ze_ds["Zdd"] = xr.DataArray(
        data=ds["Zdlog_vfov_modv_tm"]
        .sel(radar_frequencies=ds.radar_frequency, method="nearest")
        .data,
        dims=["time"],
        attrs={
            "long_name": "Disdrometer forward-modeled reflectivity",
            "units": "dBZ",
        },
    )
    # with Zdlog_vfov_measV_tm, results match with "Heraklion" codes
    Ze_ds["fallspeed_dd"] = xr.DataArray(
        data=np.nansum(
            np.nansum(ds["psd"].values, axis=2) * ds["measV"].values, axis=1
        ),
        dims=["time"],
        attrs={
            "long_name": "Average droplet fall speed seen by the disdrometer",
            "units": "dBZ",
        },
    )
    # Delta Ze
    Ze_ds["Delta_Z"] = xr.DataArray(
        data=Ze_ds["Zdcr"].data - Ze_ds["Zdd"].data.reshape((-1, 1)),
        dims=["time", "range"],
        attrs={
            "long_name": "Difference between DCR and disdrometer-modeled reflectivity",
            "units": "dBZ",
            "comment": "available at the ranges closest to those defined in the station configuration file",  # noqa E501
        },
    )

    return Ze_ds


def compute_quality_checks(ds, conf, start, end):
    if bool(ds["weather_data_avail"].values[0]) is True:
        qc_ds = processing.compute_quality_checks_weather(ds, conf, start, end)
        lgr.info("Compute QC dataset (case with weather)")
    else:
        qc_ds = processing_noweather.compute_quality_checks_noweather(
            ds, conf, start, end
        )
        lgr.info("Compute QC dataset (case without weather)")
    return qc_ds


def compute_todays_events_stats(ds, Ze_ds, conf, qc_ds, start, end):
    if bool(ds["weather_data_avail"].values[0]) is True:
        stats_ds = processing.compute_todays_events_stats_weather(
            ds, Ze_ds, conf, qc_ds, start, end
        )
        lgr.info("Compute event stats dataset (case with weather)")
    else:
        stats_ds = processing_noweather.compute_todays_events_stats_noweather(
            ds, Ze_ds, conf, qc_ds, start, end
        )
        lgr.info("Compute event stats dataset (case without weather)")
    return stats_ds


def store_outputs(ds, conf):
    start, end = rain_event_selection(ds, conf)
    Ze_ds = extract_dcr_data(ds, conf)
    qc_ds = compute_quality_checks(ds, conf, start, end)
    stats_ds = compute_todays_events_stats(ds, Ze_ds, conf, qc_ds, start, end)
    processed_ds = xr.merge([Ze_ds, qc_ds, stats_ds], combine_attrs="no_conflicts")
    output_path = "./{}_{}_processed.nc".format(
        ds.attrs["station_name"],
        pd.to_datetime(ds.time.isel(time=len(qc_ds.time) // 2).values).strftime(
            "%Y-%m-%d"
        ),
    )
    processed_ds["weather_data_avail"] = ds["weather_data_avail"]
    processed_ds.to_netcdf(
        output_path, encoding={"time": {"units": TIME_UNITS, "calendar": TIME_CALENDAR}}
    )
    return processed_ds


if __name__ == "__main__":
    test_weather = False
    test_noweather = True

    if test_weather:
        yesterday = "../../tests/data/outputs/palaiseau_2022-10-13_basta-parsivel-ws_preprocessed.nc"  # noqa E501
        today = "../../tests/data/outputs/palaiseau_2022-10-14_basta-parsivel-ws_preprocessed.nc"  # noqa E501
        tomorrow = "../../tests/data/outputs/palaiseau_2022-10-15_basta-parsivel-ws_preprocessed.nc"  # noqa E501
        conf = toml.load(
            "../../tests/data/conf/config_palaiseau_basta-parsivel-ws.toml"
        )

        ds = merge_preprocessed_data(yesterday, today, tomorrow)
        start, end = rain_event_selection(ds, conf)

        Ze_ds = extract_dcr_data(ds, conf)
        # print(Ze_ds)
        qc_ds = compute_quality_checks(ds, conf, start, end)
        events_stats_ds = processing.compute_todays_events_stats_weather(
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

    if test_noweather:
        # compare to values in events csv files used for "Heraklion" plots @ JOYCE
        yesterday = (
            "../../tests/data/outputs/juelich_2021-12-04_mira-parsivel_preprocessed.nc"
        )
        today = (
            "../../tests/data/outputs/juelich_2021-12-05_mira-parsivel_preprocessed.nc"
        )
        tomorrow = (
            "../../tests/data/outputs/juelich_2021-12-06_mira-parsivel_preprocessed.nc"
        )
        conf = toml.load("../../tests/data/conf/config_juelich_mira-parsivel.toml")

        ds = merge_preprocessed_data(yesterday, today, tomorrow)
        print(ds.attrs)
        processed_ds = store_outputs(ds, conf)

        print(processed_ds.dims)
        print(processed_ds.attrs)
        print(list(processed_ds.keys()))
        for key in processed_ds.keys():  # noqa
            # print(key, processed_ds[key].attrs)
            if processed_ds[key].attrs == {}:
                print(key)
