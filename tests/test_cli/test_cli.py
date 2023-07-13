"""Tests for `ccres_disdrometer_processing` package."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from ccres_disdrometer_processing.cli import cli

MAIN_DIR = Path(__file__).parent.parent.parent
TEST_DIR = MAIN_DIR / "tests"
TEST_INPUT = TEST_DIR / "data/inputs"
TEST_OUT_DIR = TEST_DIR / "data/outputs"
EXE = MAIN_DIR / "ccres_disdrometer_processing/cli/cli2.py"


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_command_line_interface_help():
    """Test the Help argument of CLI."""
    runner = CliRunner()
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert "Show this message and exit." in help_result.output


def test_command_line_interface():
    """Test the CLI."""
    disdro_type = "parsivel-cloudnet"
    disdro_file = f"{TEST_INPUT}/20210202_palaiseau_parsivel.nc"
    ws_type = "sirta-cloudnet"
    ws_file = f"{TEST_INPUT}/20210202_palaiseau_weather-station.nc"
    config_file = f"{TEST_INPUT}/CONFIG_test.toml"
    output_file = f"{TEST_OUT_DIR}/20210202_palaiseau_preprocessed.nc"
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [
            "--disdro-type",
            disdro_type,
            "--disdro-file",
            disdro_file,
            "--ws-type",
            ws_type,
            "--ws-file",
            ws_file,
            "--config-file",
            config_file,
            output_file,
        ],
    )
    print(result.output)
    assert result.exit_code == 0
