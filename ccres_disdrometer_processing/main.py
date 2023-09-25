"""Console script for disdrometers reflectivity calculation."""
import os

import click
import constants as constants
import create_input_files_quicklooks as input_ql
import numpy as np
import open_disdro_netcdf as disdro
import open_radar_netcdf as radar
import open_weather_netcdf as weather
import pandas as pd
import quicklooks_1event as ql_1event
import rain_event_selection as rain_events
import scattering as dcrcc
import toml
import xarray as xr

# import ccres_disdrometer_processing.ccres_disdrometer_processing.scattering as dcrcc
# import ccres_disdrometer_processing.constants as constants
# from ccres_disdrometer_processing.ccres_disdrometer_processing import (
#     open_disdro_netcdf as disdro,
# )
# from ccres_disdrometer_processing.ccres_disdrometer_processing import (
#     open_weather_netcdf as weather,
# )


DISDRO_TYPES = ["parsivel_cloudnet"]
AMS_TYPES = ["weather_station_cloudnet"]
# RADAR_TYPES = ["basta-cloudnet"]


CONFIG_FILE = "CONFIG.toml"
CONFIG_FILE_LOOP = "CONFIG_disdro_loop.toml"
CONF = True
# @click.command()
# @click.option("--config-file", type=click.Path(exists=True), required=True)


def main(config_file=CONFIG_FILE_LOOP):
    """Compute disdrometer reflectivity and merge AMS data."""
    click.echo("disdro reflectivity")
    click.echo("=" * len("disdro reflectivity"))
    click.echo("Process disdrometer data to get reflectivity")

    if CONF:
        config = toml.load(config_file)
        # print(config)

        ams_file = config["data"]["AMS_FILE"]
        disdro_file = config["data"]["DISDRO_FILE"]
        output_file = config["data"]["OUTPUT_FILE"]
        ams_type = config["data"]["AMS_TYPE"]
        disdro_type = config["data"]["DISDRO_TYPE"]

        beam_orientation = constants.BEAM_ORIENTATION
        FREQ = constants.FREQ
        E = constants.E
        E = E[0] + E[1] * 1j
        # print("e : ", E)

        axrMethod = config["methods"]["AXIS_RATIO_METHOD"]
        strMethod = config["methods"]["FALL_SPEED_METHOD"]
        mieMethod = config["methods"]["COMPUTE_MIE_METHOD"]  # pymiecoated OR pytmatrix
        normMethod = config["methods"]["NORMALIZATION_METHOD"]  # measurement OR model

    else:
        pass

    # read weather-station data
    # ---------------------------------------------------------------------------------

    if ams_type == "weather_station_cloudnet":
        ams_xr = weather.read_weather_cloudnet(ams_file)

    # read and preprocess disdrometer data
    # ---------------------------------------------------------------------------------

    if disdro_type == "parsivel_cloudnet":
        disdro_xr = disdro.read_parsivel_cloudnet(disdro_file)
        scatt = dcrcc.scattering_prop(
            disdro_xr.size_classes[0:-5],
            beam_orientation,
            FREQ,
            E,
            axrMethod,
            mieMethod=mieMethod,
        )
        F = constants.F_PARSIVEL  # m2, sampling surface
        disdro_xr = disdro.reflectivity_model(
            disdro_xr,
            scatt,
            len(disdro_xr.size_classes[0:-5]),
            F,
            FREQ,
            strMethod,
            mieMethod,
            normMethod,
        )

    final_data = xr.merge([ams_xr, disdro_xr])

    final_data.to_netcdf(output_file)


def main_loop(config_file=CONFIG_FILE_LOOP):
    if CONF:
        config = toml.load(config_file)
        # print(config)
        disdro_files = (
            config["data"]["DATA_DIR"]
            + "{}/".format(config["data"]["STATION"])
            + "disdrometer/{}{}{}_"
            + "{}_{}.nc".format(config["data"]["STATION"], config["data"]["DISDRO"])
        )
        ams_files = (
            config["data"]["DATA_DIR"]
            + "{}/".format(config["data"]["STATION"])
            + "weather-station/{}{}{}_"
            + "{}_{}.nc".format(config["data"]["STATION"], config["data"]["AMS"])
        )
        output_files = (
            config["data"]["DATA_DIR"]
            + "{}/".format(config["data"]["STATION"])
            + "disdrometer_preprocessed/{}{}{}_"
            + "{}_preprocessed.nc".format(config["data"]["STATION"])
        )
        ams_type = config["data"]["AMS_TYPE"]
        disdro_type = config["data"]["DISDRO_TYPE"]

        beam_orientation = constants.BEAM_ORIENTATION
        FREQ = constants.FREQ
        E = constants.E
        # print("e : ", E)

        axrMethod = config["methods"]["AXIS_RATIO_METHOD"]
        strMethod = config["methods"]["FALL_SPEED_METHOD"]
        mieMethod = config["methods"]["COMPUTE_MIE_METHOD"]  # pymiecoated OR pytmatrix
        normMethod = config["methods"]["NORMALIZATION_METHOD"]  # measurement OR model
    else:
        pass

    years_list = np.arange(
        config["period"]["BEGIN_DATE"][0], config["period"]["END_DATE"][0] + 1, 1
    )

    months = (
        np.arange(
            config["period"]["BEGIN_DATE"][1] - 1,
            config["period"]["END_DATE"][1] + 12 * (len(years_list) - 1),
            1,
        )
        % 12
        + 1
    )

    if len(years_list) == 1:
        years = np.array([years_list[0]] * len(months))
    if len(years_list) == 2:
        years = np.array(
            [years_list[0]] * (12 - config["period"]["BEGIN_DATE"][1] + 1)
            + [years_list[-1]] * (config["period"]["BEGIN_DATE"][1])
        )
    if len(years_list) > 2:
        years_middle = np.hstack(
            (np.tile(years_list[i], 12) for i in range(1, len(years_list) - 1))
        ).tolist()
        years = np.array(
            [years_list[0]] * (12 - config["period"]["BEGIN_DATE"][1] + 1)
            + years_middle
            + [years_list[-1]] * (config["period"]["BEGIN_DATE"][1])
        )

    for year, month in zip(years, months):
        month = "{:02d}".format(month)
        for d in range(1, 32):
            day = "{:02d}".format(d)
            print("{}/{}/{}".format(year, month, day))
            disdro_file = disdro_files.format(year, month, day)
            ams_file = ams_files.format(year, month, day)
            output_file = output_files.format(year, month, day)
            print("HELLO")
            print((os.path.exists(disdro_file)) and (os.path.exists(ams_file)))
            if (os.path.exists(disdro_file)) and (os.path.exists(ams_file)):
                # read weather-station data
                # ---------------------------------------------------------------------------------

                if ams_type == "weather_station_cloudnet":
                    ams_xr = weather.read_weather_cloudnet(ams_file)

                # read and preprocess disdrometer data
                # ---------------------------------------------------------------------------------

                if disdro_type == "parsivel_cloudnet":
                    disdro_xr = disdro.read_parsivel_cloudnet(disdro_file)
                    scatt = dcrcc.scattering_prop(
                        disdro_xr.size_classes[0:-5],
                        beam_orientation,
                        FREQ,
                        E,
                        axrMethod,
                        mieMethod=mieMethod,
                    )
                    F = constants.F_PARSIVEL  # m2, sampling surface
                    disdro_xr = disdro.reflectivity_model(
                        disdro_xr,
                        scatt,
                        len(disdro_xr.size_classes[0:-5]),
                        F,
                        FREQ,
                        strMethod,
                        mieMethod,
                        normMethod,
                    )

                final_data = xr.merge([ams_xr, disdro_xr])

                final_data.to_netcdf(output_file)


def main_loop_degrade(
    config_file=CONFIG_FILE_LOOP, add_radar=False
):  # Juelich Mira : 35GHz, not 95
    if CONF:
        config = toml.load(config_file)
        # print(config)
        disdro_files = (
            config["data"]["DATA_DIR"]
            + "{}/".format(config["data"]["STATION"])
            + "disdrometer/{}{}{}_"
            + "{}_{}.nc".format(config["data"]["STATION"], config["data"]["DISDRO"])
        )

        radar_files = (
            config["data"]["DATA_DIR"]
            + "{}/".format(config["data"]["STATION"])
            + "radar/{}{}{}_"
            + "{}_{}.nc".format(config["data"]["STATION"], config["data"]["RADAR"])
        )

        output_files = (
            config["data"]["DATA_DIR"]
            + "{}/".format(config["data"]["STATION"])
            + "disdrometer_preprocessed/{}{}{}_"
            + "{}_preprocessed_degrade.nc".format(config["data"]["STATION"])
        )
        # disdro_type = config["data"]["DISDRO_TYPE"]

        FREQ_J = constants.FREQ_35
        E = constants.E
        # print("e : ", E)

        beam_orientation = config["methods"]["BEAM_ORIENTATION"]
        axrMethod = config["methods"]["AXIS_RATIO_METHOD"]
        strMethod = config["methods"]["FALL_SPEED_METHOD"]
        mieMethod = config["methods"]["COMPUTE_MIE_METHOD"]  # pymiecoated OR pytmatrix
        normMethod = config["methods"]["NORMALIZATION_METHOD"]  # measurement OR model
    else:
        pass

    years_list = np.arange(
        config["period"]["BEGIN_DATE"][0], config["period"]["END_DATE"][0] + 1, 1
    )

    months = (
        np.arange(
            config["period"]["BEGIN_DATE"][1] - 1,
            config["period"]["END_DATE"][1] + 12 * (len(years_list) - 1),
            1,
        )
        % 12
        + 1
    )

    if len(years_list) == 1:
        years = np.array([years_list[0]] * len(months))
    if len(years_list) == 2:
        years = np.array(
            [years_list[0]] * (12 - config["period"]["BEGIN_DATE"][1] + 1)
            + [years_list[-1]] * (config["period"]["BEGIN_DATE"][1])
        )
    if len(years_list) > 2:
        tup = [np.tile(years_list[i], 12) for i in range(1, len(years_list) - 1)]
        print(tup, type(tup))
        years_middle = np.hstack(tup).tolist()
        years = np.array(
            [years_list[0]] * (12 - config["period"]["BEGIN_DATE"][1] + 1)
            + years_middle
            + [years_list[-1]] * (config["period"]["BEGIN_DATE"][1])
        )

    for year, month in zip(years, months):
        month = "{:02d}".format(month)
        for d in range(1, 32):
            day = "{:02d}".format(d)
            print("{}/{}/{}".format(year, month, day))
            disdro_file = disdro_files.format(year, month, day)
            radar_file = radar_files.format(year, month, day)
            output_file = output_files.format(year, month, day)
            print(
                disdro_file, (os.path.exists(disdro_file))
            )  # , (os.path.exists(radar_file)))
            if os.path.exists(disdro_file):  # and os.path.exists(radar_file):
                # read radar data
                # ---------------------------------------------------------------------------------
                if add_radar:
                    radar_xr = radar.read_radar_cloudnet(radar_file)

                # read and preprocess disdrometer data
                # ---------------------------------------------------------------------------------

                disdro_xr = disdro.read_parsivel_cloudnet_choice(disdro_file)

                scatt = dcrcc.scattering_prop(
                    disdro_xr.size_classes[0:-5],
                    beam_orientation,
                    # radar_xr["lambda"].data,
                    FREQ_J,
                    E,
                    axrMethod,
                    mieMethod=mieMethod,
                )
                disdro_xr = disdro.reflectivity_model(
                    disdro_xr,
                    scatt,
                    len(disdro_xr.size_classes[0:-5]),
                    # radar_xr["lambda"].data,
                    FREQ_J,
                    strMethod,
                    mieMethod,
                    normMethod,
                )

                final_data = xr.merge([disdro_xr, radar_xr])
                final_data.to_netcdf(output_file)

                # disdro_xr.to_netcdf(output_file)


def main_loop_quicklooks(config_file=CONFIG_FILE_LOOP):
    """
    1 = compute rain event selection over the defined period
    2 = parcourir les dates d'événement et créer les fichiers radar,
    disdro(prepro) et WS à ces dates
    for start, end in zip(start_events, end_events) :
    get_data_event(config.DATA_DIR, config.STATION, start, end,
    thresholds=TIMESTAMP_THRESHOLDS + EVENT_THRESHOLDS)
    quicklooks(weather,dcr,disdro,output_path,
    thresholds=TIMESTAMP_THRESHOLDS + EVENT_THRESHOLDS,)
    3 = appliquer la méthode quicklooks en boucle à ces fichiers; stocker à un endroit contrôlé
    """
    config = toml.load(config_file)
    data_dir_root = config["data"]["DATA_DIR"] + "{}/".format(config["data"]["STATION"])
    weather_station_data_path = data_dir_root + "weather-station"
    ql_output_path = config["data"]["QUICKLOOKS_DIR"] + "{}/".format(
        config["data"]["STATION"]
    )
    main_wind_direction = config["data"]["MAIN_WIND_DIR"]

    events = rain_events.selection(
        station=config["data"]["STATION"],
        ws_data_dir=weather_station_data_path,
        db_dir=config["data"]["BDD_RAIN_EVENTS_PATH"],
    )

    time_inf = pd.Timestamp(
        year=config["period"]["BEGIN_DATE"][0],
        month=config["period"]["BEGIN_DATE"][1],
        day=1,
    )
    time_sup = pd.Timestamp(
        year=config["period"]["END_DATE"][0],
        month=config["period"]["END_DATE"][1],
        day=(config["period"]["END_DATE"][1] == 2) * 28
        + (config["period"]["END_DATE"][1] != 2) * 30,
    )

    events_timeperiod = events.iloc[
        np.where(
            (events["Start_time"] >= time_inf) & (events["Start_time"] <= time_sup)
        )
    ]

    for start, end in zip(
        events_timeperiod["Start_time"], events_timeperiod["End_time"]
    ):
        start = start.to_pydatetime()
        end = end.to_pydatetime()
        weather, dcr, disdro = input_ql.get_data_event(
            data_dir_root, start, end, main_wind_dir=main_wind_direction
        )
        print(list(weather.keys()))
        # plt.figure()
        # plt.plot(weather.time, weather["rain_sum"])
        # plt.plot(weather.time, weather.QF_acc)
        # plt.show()
        q = ql_1event.quicklooks(weather, dcr, disdro, ql_output_path)
        print(q)
    return


if __name__ == "__main__":
    conf_lindenberg = "CONFIG_LINDENBERG_LOOP_DEGRADE.toml"
    conf_juelich = "CONFIG_disdro_loop.toml"
    main_loop_degrade(config_file=conf_juelich)
