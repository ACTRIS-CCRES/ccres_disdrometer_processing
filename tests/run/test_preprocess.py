"""Tests preprocessing."""

import pytest
from click.testing import CliRunner

from ccres_disdrometer_processing.cli import cli
from tests import utils


@pytest.fixture
def test_case(test_data_preprocessing):
    yield from test_data_preprocessing


def test_run(test_case, data_input_dir, data_conf_dir, data_out_dir) -> None:
    """Test the preprocessing for a specific test case."""
    site = test_case["site"]
    date = test_case["date"]
    radar = test_case["radar"]
    disdro = test_case["disdro"]
    has_meteo = test_case["meteo-available"]
    meteo = test_case["meteo"]
    conf = test_case["config_file"]

    # get the data if needed
    # ---------------------------------------------------------------------------------
    # radar
    radar_file = utils.get_file_from_cloudnet(site, date, radar, data_input_dir)
    # disdro
    disdro_file = utils.get_file_from_cloudnet(site, date, disdro, data_input_dir)
    # meteo
    if test_case["meteo-available"]:
        meteo_file = utils.get_file_from_cloudnet(site, date, meteo, data_input_dir)

    # other parameters
    # ---------------------------------------------------------------------------------
    # conf
    conf = data_conf_dir / conf
    # output
    output_file = data_out_dir / f"{site}_{date}_preprocessed.nc"

    # run the preprocessing
    # ---------------------------------------------------------------------------------
    # required args
    args = [
        "--disdro-file",
        str(disdro_file),
        "--radar-file",
        str(radar_file),
        "--config-file",
        str(conf),
    ]
    # add meteo if available
    if has_meteo:
        args += [
            "--ws-file",
            str(meteo_file),
        ]

    args += [str(output_file)]

    runner = CliRunner()
    result = runner.invoke(
        cli.preprocess,
        args,
        catch_exceptions=False,
    )

    assert result.exit_code == 0
