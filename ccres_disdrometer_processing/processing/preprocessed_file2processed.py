"""Apply the processing from daily preprocessed files.

Input : Daily preprocessed files at days D and D-1
Output : Daily processed file for day D

"""

import datetime
import logging

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml
import xarray as xr

from ccres_disdrometer_processing.__init__ import __version__, script_name
from ccres_disdrometer_processing.logger import LogLevels, init_logger
from ccres_disdrometer_processing.processing import (
    preprocessed_file2processed_noweather as processing_noweather,
)
from ccres_disdrometer_processing.processing import (
    preprocessed_file2processed_weather as processing,
)

ISO_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
TIME_UNITS = "seconds since 2000-01-01T00:00:00.0Z"
TIME_CALENDAR = "standard"
QC_FILL_VALUE = -9

lgr = logging.getLogger(__name__)


def merge_preprocessed_data(yesterday, today, tomorrow):
    lgr.info("Beginning rain event selection")

    # TODO: add log for missing files
    list_files = [yesterday, today, tomorrow]
    files_provided = [f is not None for f in list_files]
    list_available_files = [f for f in list_files if f is not None]
    if list_files != list_available_files:
        lgr.info("At least one daily file is missing, preliminary processing")

    tmp_data = []
    for file in list_available_files:
        tmp_data.append(xr.open_dataset(file))

    ds = xr.concat(tmp_data, dim="time")
    return ds, files_provided


def rain_event_selection(ds, conf, no_meteo):
    if bool(ds["weather_data_avail"].values[0]) is False or no_meteo is True:
        lgr.info(
            "Rain event selection (no rain gauge data, disdrometer precipitation is used)"  # noqa
        )
        start, end = processing_noweather.rain_event_selection_noweather(ds, conf)
    else:
        lgr.info("Rain event selection (from rain gauge data)")
        start, end = processing.rain_event_selection_weather(ds, conf)
    return start, end


def extract_dcr_data(ds, conf):
    # Extract DCR Ze at 3/4 relevant gates, extract DD Ze, compute Delta Ze
    # Get Doppler velocity at relevant gates, compute avg disdrometer fall speed(t)
    ranges_to_keep = conf["instrument_parameters"]["PROCESSING_RANGES"]
    Ze_ds = xr.Dataset(
        coords=dict(
            time=(["time"], ds.time.data),
            range=(["range"], np.array(ranges_to_keep, dtype=np.single)),
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
        .data.astype(np.single),
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
        ).astype(np.single),
        dims=["time"],
        attrs={
            "long_name": "Average droplet fall speed seen by the disdrometer",
            "units": "dBZ",
        },
    )
    # Delta Ze
    Ze_ds["Delta_Z"] = xr.DataArray(
        data=Ze_ds["Zdcr"].data - Ze_ds["Zdd"].data.reshape((-1, 1)).astype(np.single),
        dims=["time", "range"],
        attrs={
            "long_name": "Difference between DCR and disdrometer-modeled reflectivity",
            "units": "dBZ",
            "comment": "available at the ranges closest to those defined in the station configuration file",  # noqa E501
        },
    )

    return Ze_ds


def compute_quality_checks(ds, conf, start, end, no_meteo):
    if bool(ds["weather_data_avail"].values[0]) is False or no_meteo is True:
        qc_ds = processing_noweather.compute_quality_checks_noweather(
            ds, conf, start, end
        )
        lgr.info("Compute QC dataset (case without weather)")
    else:
        data_avail = ds.time.isel({"time": np.where(np.isfinite(ds["ta"]))[0]}).values
        ams_time_sampling = (data_avail[1] - data_avail[0]) / np.timedelta64(1, "m")
        lgr.info(f"AMS time sampling : {ams_time_sampling:.0f} minute(s)")
        if ams_time_sampling < 1.5:
            lgr.info("AMS data available at 1mn frequency")
            qc_ds = processing.compute_quality_checks_weather(ds, conf, start, end)
        else:
            qc_ds = processing.compute_quality_checks_weather_low_sampling(
                ds, conf, start, end
            )
            lgr.info("AMS data available at a frequency > 1mn")
        lgr.info("Compute QC dataset (case with weather)")
    return qc_ds


def compute_todays_events_stats(
    ds, Ze_ds, conf, qc_ds, start, end, no_meteo, day_today
):
    if bool(ds["weather_data_avail"].values[0]) is False or no_meteo is True:
        stats_ds = processing_noweather.compute_todays_events_stats_noweather(
            ds, Ze_ds, conf, qc_ds, start, end, day_today
        )
        lgr.info("Compute event stats dataset (case without weather)")
    else:
        stats_ds = processing.compute_todays_events_stats_weather(
            ds, Ze_ds, conf, qc_ds, start, end, day_today
        )
        lgr.info("Compute event stats dataset (case with weather)")
    return stats_ds


def add_attributes(processed_ds, preprocessed_ds):
    # Add global attributes specified in the file format
    keys_from_preprocessed = [
        "year",
        "month",
        "day",
        "location",
        "disdrometer_source",
        "disdrometer_pid",
        "radar_source",
        "radar_pid",
        "station_name",
        "axis_ratioMethod",
        "fallspeedFormula",
    ]
    for key in keys_from_preprocessed:
        processed_ds.attrs[key] = preprocessed_ds.attrs[key]

    processed_ds.attrs["title"] = (
        f"CCRES processing output file for Doppler cloud radar stability monitoring with disdrometer at {processed_ds.attrs['location']} site"  # noqa E501
    )
    processed_ds.attrs["summary"] = (
        f"Significant rain events are identified, and statistics of reflectivity differences between DCR and disdrometer-modeled data are computed at relevant radar ranges over each rain event period, after applying a filter based on several quality checks."  # noqa E501
    )

    for key in [
        "keywords",
        "keywords_vocabulary",
        "Conventions",
        "id",
        "naming_authority",
    ]:
        processed_ds.attrs[key] = preprocessed_ds.attrs[key]

    date_created = datetime.datetime.utcnow().strftime(ISO_DATE_FORMAT)
    processed_ds.attrs["history"] = (
        f"created on {date_created} by {script_name}, v{__version__}"
    )
    processed_ds.attrs["date_created"] = date_created
    weather_str = ""
    if processed_ds.weather_data_used:
        weather_str = " and AMS"
    processed_ds.attrs["source"] = (
        f"surface observation from {processed_ds.radar_source} DCR, {processed_ds.disdrometer_source} disdrometer{weather_str}, processed by CloudNet"  # noqa
    )
    processed_ds.attrs["processing_level"] = "2b"

    for key in [
        "comment",
        "acknowledgement",
        "license",
        "standard_name_vocabulary",
        "creator_name",
        "creator_email",
        "creator_url",
        "creator_type",
        "creator_institution",
        "project",
        "publisher_name",
        "publisher_email",
        "publisher_url",
        "publisher_type",
        "publisher_institution",
        "contributor_name",
        "contributor_role",
    ]:
        processed_ds.attrs[key] = preprocessed_ds.attrs[key]

    for key in [
        "geospatial_bounds",
        "geospatial_bounds_crs",
        "geospatial_bounds_vertical_crs",
        "geospatial_lat_min",
        "geospatial_lat_max",
        "geospatial_lat_units",
        "geospatial_lat_resolution",
        "geospatial_lon_min",
        "geospatial_lon_max",
        "geospatial_lon_units",
        "geospatial_lon_resolution",
        "geospatial_vertical_min",
        "geospatial_vertical_max",
        "geospatial_vertical_units",
        "geospatial_vertical_resolution",
        "geospatial_vertical_positive",
        "time_coverage_start",
        "time_coverage_end",
        "time_coverage_duration",
        "time_coverage_resolution",
        "program",
    ]:
        processed_ds.attrs[key] = preprocessed_ds.attrs[key]

    processed_ds.attrs["date_modified"] = date_created
    processed_ds.attrs["date_issued"] = (
        date_created  # file made available immediately to the users after creation
    )
    processed_ds.attrs["date_metadata_modified"] = (
        ""  # will be set when everything will be of ; modify it if some fields evolve
    )
    processed_ds.attrs["product_version"] = __version__

    for key in [
        "platform",
        "platform_vocabulary",
        "instrument",
        "instrument_vocabulary",
        "cdm_data_type",
        "metadata_link",
        "references",
    ]:
        processed_ds.attrs[key] = preprocessed_ds.attrs[key]

    return


def process(yesterday, today, tomorrow, conf, output_file, no_meteo, verbosity):
    """Use to build command line interface for processing step."""
    log_level = LogLevels.get_by_verbosity_count(verbosity)
    init_logger(log_level)
    click.echo("CCRES disdrometer processing : CLI")

    conf = toml.load(conf)
    ds, files_provided = merge_preprocessed_data(yesterday, today, tomorrow)

    day_today = pd.to_datetime(xr.open_dataset(today)["time"].time.values[0]).day

    if bool(ds["weather_data_avail"].values[0]) is False or no_meteo is True:
        click.echo("Downgraded mode (no weather data is used)")

    start, end = rain_event_selection(ds, conf, no_meteo)
    Ze_ds = extract_dcr_data(ds, conf)
    qc_ds = compute_quality_checks(ds, conf, start, end, no_meteo)
    stats_ds = compute_todays_events_stats(
        ds, Ze_ds, conf, qc_ds, start, end, no_meteo, day_today
    )
    lgr.info("Merging Ze data, QC dataset and event stats dataset")
    processed_ds = xr.merge([Ze_ds, qc_ds, stats_ds], combine_attrs="no_conflicts")
    # Select only timesteps from the day to process
    today_ds = xr.open_dataset(today)
    lgr.info("Extracting the data from the day to process")
    processed_ds = processed_ds.sel(
        {"time": slice(today_ds.time.values[0], today_ds.time.values[-1])}
    )
    lgr.info("Filling attributes")
    # get variable for weather data availability from prepro file
    if not no_meteo:
        processed_ds["weather_data_used"] = (
            ds["weather_data_avail"].values[0].astype("i2")
        )
    else:
        processed_ds["weather_data_used"] = np.array([0])[0].astype("i2")
    processed_ds["weather_data_used"].attrs = {
        "long_name": "use of weather data for processing",
        "flag_values": np.array([0, 1]).astype("i2"),
        "flag_meanings": "yes no ",
    }
    # set attributes
    add_attributes(processed_ds, ds)
    str_files_provided = ""
    daily_files = ["D-1", "D", "D+1"]
    for i in range(len(files_provided)):
        if files_provided[i] is True:
            if str_files_provided != "":
                str_files_provided += " "
            str_files_provided += daily_files[i]
    processed_ds.attrs["files_provided"] = str_files_provided
    processed_ds.attrs["number_files_provided"] = np.sum(np.array(files_provided) * 1)

    processed_ds.rename_dims({"events": "event"})  # rename dimension
    # save to netCDF
    lgr.info("Saving to netCDF")
    processed_ds.to_netcdf(
        output_file,
        encoding={
            "time": {"units": TIME_UNITS, "calendar": TIME_CALENDAR},
            "start_event": {"units": TIME_UNITS, "calendar": TIME_CALENDAR},
            "end_event": {"units": TIME_UNITS, "calendar": TIME_CALENDAR},
            "ams_cp_since_event_begin": {"_FillValue": np.nan},
            "disdro_cp_since_event_begin": {"_FillValue": np.nan},
            "QC_ta": {"_FillValue": QC_FILL_VALUE},
            "QC_pr": {"_FillValue": QC_FILL_VALUE},
            "QC_vdsd_t": {"_FillValue": QC_FILL_VALUE},
            "QF_rainfall_amount": {"_FillValue": QC_FILL_VALUE},
            "QC_ws": {"_FillValue": QC_FILL_VALUE},
            "QC_wd": {"_FillValue": QC_FILL_VALUE},
            "QF_rg_dd": {"_FillValue": QC_FILL_VALUE},
            "QC_ta_ratio": {"_FillValue": np.nan},
            "QC_ws_ratio": {"_FillValue": np.nan},
            "QC_wd_ratio": {"_FillValue": np.nan},
            "QF_rg_dd_event": {"_FillValue": QC_FILL_VALUE},
        },
    )
    lgr.info("Processing : success")
    return


if __name__ == "__main__":
    test_weather = False
    test_weather_downgraded = False
    test_noweather = False
    test_lindenberg_10mn = False
    test_no_meteo = (
        False  # downgraded without weather even when available in prepro files
    )
    test_lindenberg_low_sampling = True

    if test_lindenberg_10mn:
        yesterday = None
        today = "../../tests/data/outputs/lindenberg_2023-09-22_rpg-parsivel_preprocessed.nc"  # noqa E501
        tomorrow = None  # noqa E501
        ds, files_provided = merge_preprocessed_data(yesterday, today, tomorrow)
        conf = "../../tests/data/conf/config_lindenberg_rpg-parsivel.toml"
        start, end = processing.rain_event_selection_weather(ds, toml.load(conf))
        print(start, end)

    if test_weather_downgraded:
        yesterday = None
        today = "../../tests/data/outputs/lindenberg_2023-09-22_rpg-parsivel_preprocessed.nc"  # noqa E501
        tomorrow = None  # noqa E501
        conf = "../../tests/data/conf/config_lindenberg_rpg-parsivel.toml"

        ds, files_provided = merge_preprocessed_data(yesterday, today, tomorrow)
        output_file = "./{}_{}_processed_downgraded.nc".format(
            ds.attrs["location"].lower(),
            pd.to_datetime(ds.time.isel(time=ds.time.size // 2).values).strftime(
                "%Y-%m-%d"
            ),
        )
        process(yesterday, today, tomorrow, conf, output_file, 1)
        processed_ds = xr.open_dataset(output_file)
        print(processed_ds["QC_overall"].attrs)
        print(processed_ds.dims)

    if test_weather:
        yesterday = "../../tests/data/outputs/palaiseau_2022-10-13_basta-parsivel-ws_preprocessed.nc"  # noqa E501
        today = "../../tests/data/outputs/palaiseau_2022-10-14_basta-parsivel-ws_preprocessed.nc"  # noqa E501
        tomorrow = "../../tests/data/outputs/palaiseau_2022-10-15_basta-parsivel-ws_preprocessed.nc"  # noqa E501
        conf = "../../tests/data/conf/config_palaiseau_basta-parsivel-ws.toml"

        ds, files_provided = merge_preprocessed_data(yesterday, today, tomorrow)
        output_file = "./{}_{}_processed.nc".format(
            ds.attrs["station_name"],
            pd.to_datetime(ds.time.isel(time=ds.time.size // 2).values).strftime(
                "%Y-%m-%d"
            ),
        )
        process(
            yesterday, today, tomorrow, conf, output_file, no_meteo=False, verbosity=1
        )
        processed_ds = xr.open_dataset(output_file)
        QCwd = processed_ds.QC_wd.values
        print(len(QCwd[np.where(QCwd == 1)]))
        for key in list(processed_ds.keys()):
            if "events" in processed_ds[key].dims:
                print(key)

        plt.figure()
        plt.plot(
            processed_ds.time,
            processed_ds.ams_cp_since_event_begin.values,
            color="blue",
            label="ams rainfall amount",
        )
        plt.plot(
            processed_ds.time,
            processed_ds.disdro_cp_since_event_begin.values,
            color="red",
            label="disdro rainfall amount",
        )
        plt.plot(
            processed_ds.time,
            processed_ds.QC_pr.values * 8,
            color="green",
            label="QC pr",
        )
        plt.legend()
        plt.savefig("./plot_diagnostic_preprocessing.png", dpi=300)
        plt.close()

        # processed_ds = xr.open_dataset("./JOYCE_2021-12-04_processed.nc")

        # ds, files_provided = merge_preprocessed_data(yesterday, today, tomorrow)
        # start, end = rain_event_selection(ds, conf)

        # Ze_ds = extract_dcr_data(ds, conf)
        # qc_ds = compute_quality_checks(ds, conf, start, end)
        # events_stats_ds = processing.compute_todays_events_stats_weather(
        #     Ze_ds, conf, qc_ds, start, end
        # )

        # plot = False
        # if plot:
        #     plt.figure()
        #     plt.plot(
        #         qc_ds.time,
        #         qc_ds.ams_cum_since_event_begin.values,
        #         color="blue",
        #         label="ams rainfall amount",
        #     )
        #     plt.plot(
        #         qc_ds.time,
        #         qc_ds.disdro_cum_since_event_begin.values,
        #         color="red",
        #         label="disdro rainfall amount",
        #     )
        #     plt.legend()
        #     plt.savefig("./plot_diagnostic_preprocessing.png", dpi=300)
        #     plt.close()

        #     plt.figure()
        #     # plt.plot(qc_ds.time, qc_ds.QC_ta, label="ta", alpha=0.4)
        #     # plt.plot(qc_ds.time, qc_ds.QC_ws, label="ws", alpha=0.4)
        #     plt.plot(qc_ds.time, 225 + 10 * qc_ds.QC_wd, label="qc_wd", alpha=1)
        #     # plt.plot(qc_ds.time, qc_ds.QC_pr, label="pr", alpha=0.4)
        #     # plt.plot(qc_ds.time, qc_ds.QC_vdsd_t, label="vd", alpha=0.4)
        #     # plt.plot(qc_ds.time, qc_ds.QC_overall, label="overall")
        #     # plt.plot(ds.time, ds.ws)
        #     plt.plot(ds.time, ds.wd)
        #     plt.axhline(y=225, alpha=0.3)
        #     plt.xlim(left=start[0], right=end[0])
        #     plt.legend()
        #     plt.savefig("./plot_diagnostic_preprocessing2.png", dpi=300)
        #     plt.close()

    if test_noweather:
        # compare to values in events csv files used for "Heraklion" plots @ JOYCE
        yesterday = (
            "../../tests/data/outputs/juelich_2021-12-03_mira-parsivel_preprocessed.nc"
        )
        today = (
            "../../tests/data/outputs/juelich_2021-12-04_mira-parsivel_preprocessed.nc"
        )
        tomorrow = (
            "../../tests/data/outputs/juelich_2021-12-05_mira-parsivel_preprocessed.nc"
        )
        conf = "../../tests/data/conf/config_juelich_mira-parsivel.toml"

        ds, files_provided = merge_preprocessed_data(yesterday, today, tomorrow)
        output_file = "./{}_{}_processed_scipy.nc".format(
            ds.attrs["station_name"],
            pd.to_datetime(ds.time.isel(time=ds.time.size // 2).values).strftime(
                "%Y-%m-%d"
            ),
        )
        process(
            yesterday, today, tomorrow, conf, output_file, no_meteo=False, verbosity=1
        )

        processed_ds = xr.open_dataset("./JOYCE_2021-12-04_processed_scipy.nc")
        # print(processed_ds.attrs)
        print(processed_ds.dims)
        print(processed_ds.time.values[[0, -1]])
        print(list(processed_ds.keys()))
        # for key in processed_ds.keys():  # noqa
        #     print(key, processed_ds[key].dims)
        # if processed_ds[key].attrs == {}:
        #     print(key)
        # print(
        #     processed_ds.nb_dz_computable_pts.values,
        #     processed_ds.good_points_number.values,
        # )

    if test_lindenberg_low_sampling:
        print("Test Lindenberg low sampling")
        yesterday = None
        today = "../../tests/data/outputs/lindenberg_2023-09-22_rpg-parsivel_preprocessed.nc"  # noqa E501
        tomorrow = None  # noqa E501
        conf = "../../tests/data/conf/config_lindenberg_rpg-parsivel.toml"
        output_file = "./Lindenberg_low_sampling_test.nc"
        process(
            yesterday, today, tomorrow, conf, output_file, no_meteo=False, verbosity=1
        )
