"""Tests the preprocessing with a weather-station file lacking variables."""

import numpy as np
import pandas as pd
import xarray as xr
from click.testing import CliRunner

from ccres_disdrometer_processing import utils
from ccres_disdrometer_processing.cli import cli
from ccres_disdrometer_processing.open_weather_netcdf import (
    WEATHER_VARIABLES,
    missing_weather_variables,
)

# palaiseau case with a complete weather file, from which wind_speed is removed
CASE = {
    "site": "palaiseau",
    "date": "2024-01-02",
    "radar": ("basta", "https://hdl.handle.net/21.12132/3.643b7b5b43814e6f"),
    "disdro": ("parsivel", "https://hdl.handle.net/21.12132/3.7e13f3f243854ae8"),
    "meteo": ("weather-station", "https://hdl.handle.net/21.12132/3.739041931dac4de5"),
    "config_file": "config_palaiseau_basta-parsivel-ws.toml",
}


def test_missing_weather_variables(tmp_path):
    time = pd.date_range("2024-01-02", periods=3, freq="10min")
    present = [v for v in WEATHER_VARIABLES if v not in ("wind_speed", "rainfall_rate")]
    ds = xr.Dataset({v: ("time", np.zeros(3)) for v in present}, coords={"time": time})
    ds.to_netcdf(tmp_path / "weather.nc")
    assert missing_weather_variables(tmp_path / "weather.nc") == [
        "wind_speed",
        "rainfall_rate",
    ]


def test_preprocess_ignores_incomplete_weather_file(
    tmp_path, data_input_dir, data_conf_dir
):
    site, date = CASE["site"], CASE["date"]
    files = {
        key: utils.get_file_from_cloudnet(site, date, *CASE[key], data_input_dir)
        for key in ("radar", "disdro", "meteo")
    }
    incomplete_ws = tmp_path / "weather_without_wind_speed.nc"
    with xr.open_dataset(files["meteo"]) as ws:
        ws.drop_vars("wind_speed").to_netcdf(incomplete_ws)

    output = tmp_path / "preprocessed.nc"
    args = [
        "--disdro-file",
        str(files["disdro"]),
        "--radar-file",
        str(files["radar"]),
        "--ws-file",
        str(incomplete_ws),
        "--config-file",
        str(data_conf_dir / CASE["config_file"]),
        str(output),
    ]
    result = CliRunner().invoke(cli.preprocess, args, catch_exceptions=False)

    assert result.exit_code == 0
    assert "lacking wind_speed" in result.output
    with xr.open_dataset(output) as ds:
        assert int(ds["weather_data_avail"]) == 0
        assert "wind_speed" in ds["weather_data_avail"].attrs["comment"]
        assert "ta" not in ds
