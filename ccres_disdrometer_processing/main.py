"""Console script for disdrometers reflectivity calculation."""
import click
import toml
import xarray as xr

import ccres_disdrometer_processing.ccres_disdrometer_processing.scattering as dcrcc
import ccres_disdrometer_processing.constants as constants
from ccres_disdrometer_processing.ccres_disdrometer_processing import (
    open_disdro_netcdf as disdro,
)
from ccres_disdrometer_processing.ccres_disdrometer_processing import (
    open_weather_netcdf as weather,
)

DISDRO_TYPES = ["parsivel_cloudnet"]
AMS_TYPES = ["weather_station_cloudnet"]
# RADAR_TYPES = ["basta-cloudnet"]


CONFIG_FILE = "CONFIG.toml"
CONF = True
# @click.command()
# @click.option("--config-file", type=click.Path(exists=True), required=True)


def main(config_file=CONFIG_FILE):
    """Compute disdrometer reflectivity and merge AMS and 95GHz radar data."""
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


if __name__ == "__main__":
    main(CONFIG_FILE)
