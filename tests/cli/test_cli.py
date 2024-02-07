"""Tests for `ccres_disdrometer_processing` package."""

from pathlib import Path

from click.testing import CliRunner

from ccres_disdrometer_processing.cli import cli

MAIN_DIR = Path(__file__).parent.parent.parent
TEST_DIR = MAIN_DIR / "tests"
TEST_INPUT = TEST_DIR / "data/inputs"
TEST_OUT_DIR = TEST_DIR / "data/outputs"
EXE = MAIN_DIR / "ccres_disdrometer_processing/cli/cli2.py"


def test_preprocess_interface_help() -> None:
    """Test the Help argument of CLI."""
    runner = CliRunner()
    help_result = runner.invoke(cli.preprocess, ["--help"])
    assert help_result.exit_code == 0
    assert "Show this message and exit." in help_result.output


def test_preprocess_interface() -> None:
    """Test the CLI."""
<<<<<<< HEAD:tests/test_cli/test_cli.py
    disdro_file = f"{TEST_INPUT}/20210202_palaiseau_parsivel.nc"
=======
    # disdro_type = "parsivel-cloudnet"
    disdro_file = f"{TEST_INPUT}/20210202_palaiseau_parsivel.nc"
    # ws_type = "sirta-cloudnet"
>>>>>>> 4dc8483 (rename test sub dir for uniformity):tests/cli/test_cli.py
    ws_file = f"{TEST_INPUT}/20210202_palaiseau_weather-station.nc"
    radar_file = f"{TEST_INPUT}/20210202_palaiseau_basta.nc"
    config_file = f"{TEST_INPUT}/CONFIG_test.toml"
    output_file = f"{TEST_OUT_DIR}/20210202_palaiseau_preprocessed_v2311.nc"
    runner = CliRunner()
    result = runner.invoke(
        cli.preprocess,
        [
            "--disdro-file",
            disdro_file,
            "--ws-file",
            ws_file,
            "--radar-file",
            radar_file,
            "--config-file",
            config_file,
            output_file,
        ],
    )
    print(result.output)
    assert result.exit_code == 0
