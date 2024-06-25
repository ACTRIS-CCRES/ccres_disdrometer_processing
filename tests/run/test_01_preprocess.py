"""Tests preprocessing."""

from click.testing import CliRunner

from ccres_disdrometer_processing import utils
from ccres_disdrometer_processing.cli import cli

# @pytest.fixture(params=test_data_preprocessing)
# def test_case(test_data_preprocessing):
#     yield from test_data_preprocessing


def test_run(
    test_data_preprocessing, data_input_dir, data_conf_dir, data_out_dir
) -> None:
    """Test the preprocessing for a specific test case."""
    site = test_data_preprocessing["site"]
    date = test_data_preprocessing["date"]
    radar = test_data_preprocessing["radar"]
    radar_pid = test_data_preprocessing["radar-pid"]
    disdro = test_data_preprocessing["disdro"]
    disdro_pid = test_data_preprocessing["disdro-pid"]
    has_meteo = test_data_preprocessing["meteo-available"]
    meteo = test_data_preprocessing["meteo"]
    conf = test_data_preprocessing["config_file"]
    output_file = data_out_dir / test_data_preprocessing["output"]["preprocess"]

    # get the data if needed
    # ---------------------------------------------------------------------------------
    # radar
    radar_file = utils.get_file_from_cloudnet(
        site, date, radar, radar_pid, data_input_dir
    )
    # disdro
    disdro_file = utils.get_file_from_cloudnet(
        site, date, disdro, disdro_pid, data_input_dir
    )
    # meteo
    if test_data_preprocessing["meteo-available"]:
        meteo_pid = test_data_preprocessing["meteo-pid"]
        meteo_file = utils.get_file_from_cloudnet(
            site, date, meteo, meteo_pid, data_input_dir
        )

    # other parameters
    # ---------------------------------------------------------------------------------
    # conf
    conf = data_conf_dir / conf

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
