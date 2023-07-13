"""Console script for ccres_disdrometer_processing."""

import sys
from pathlib import Path

import click
import toml
import xarray as xr

sys.path.append(str(Path(__file__).parent.parent.parent))

import ccres_disdrometer_processing.OPEN_DISDRO_NETCDF as disdro
import ccres_disdrometer_processing.OPEN_WEATHER_NETCDF as weather
import ccres_disdrometer_processing.SCATTERING as scattering
from ccres_disdrometer_processing import constants

DISDRO_TYPES = ["parsivel-cloudnet"]
WS_TYPES = ["sirta-cloudnet"]
RADAR_TYPES = ["basta-cloudnet"]  # sera vu plus tard


@click.command()
@click.option("--disdro-type", type=click.Choice(DISDRO_TYPES), required=True)
@click.option("--disdro-file", type=click.Path(exists=True), required=True)
@click.option("--ws-type", type=click.Choice(WS_TYPES), required=True)
@click.option("--ws-file", type=click.Path(exists=True), required=True)
@click.option("--config-file", type=click.Path(exists=True), required=True)
@click.argument("output-file", type=click.Path())
def main(disdro_type, disdro_file, ws_type, ws_file, config_file, output_file):
    """Command line interface for ccres_disdrometer_processing."""
    click.echo("CCRES disdrometer preprocessing")

    config = toml.load(config_file)

    beam_orientation = constants.BEAM_ORIENTATION
    freq = constants.f
    e = constants.e
    e = e[0] + e[1] * 1j

    axrMethod = config["methods"]["AXIS_RATIO_METHOD"]
    strMethod = config["methods"]["FALL_SPEED_METHOD"]
    mieMethod = config["methods"]["COMPUTE_MIE_METHOD"]  # pymiecoated OR pytmatrix
    normMethod = config["methods"]["NORMALIZATION_METHOD"]  # measurement OR model

    # read weather-station data
    # ---------------------------------------------------------------------------------
    if ws_type == "sirta-cloudnet":
        weather_xr = weather.read_weather_cloudnet(ws_file)

    # read and preprocess disdrometer data
    # ---------------------------------------------------------------------------------

    if disdro_type == "parsivel-cloudnet":
        disdro_xr = disdro.read_parsivel_cloudnet(disdro_file)
        scatt = scattering.scattering_prop(
            disdro_xr.size_classes[0:-5],
            beam_orientation,
            freq,
            e,
            axrMethod=axrMethod,
            mieMethod=mieMethod,
        )
        F = constants.F_parsivel
        disdro_xr = disdro.reflectivity_model(
            disdro_xr,
            scatt,
            len(disdro_xr.size_classes[0:-5]),
            F,
            freq,
            strMethod=strMethod,
            mieMethod=mieMethod,
            normMethod=normMethod,
        )

    final_data = xr.merge([weather_xr, disdro_xr])

    final_data.to_netcdf(output_file)

    return


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
