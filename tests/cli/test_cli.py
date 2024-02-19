"""Tests for `ccres_disdrometer_processing` package."""

from click.testing import CliRunner

from ccres_disdrometer_processing.cli import cli


def test_preprocess_interface_help() -> None:
    """Test the Help argument of CLI."""
    runner = CliRunner()
    help_result = runner.invoke(cli.preprocess, ["--help"])
    assert help_result.exit_code == 0
    assert "Show this message and exit." in help_result.output
