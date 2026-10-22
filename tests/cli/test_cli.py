"""Tests for `ccres_disdrometer_processing` package."""

from click.testing import CliRunner

from ccres_disdrometer_processing.cli import cli


def test_preprocess_interface_help() -> None:
    """Test the Help argument of CLI."""
    runner = CliRunner()
    help_result = runner.invoke(cli.preprocess, ["--help"])
    assert help_result.exit_code == 0
    assert "Show this message and exit." in help_result.output


def test_process_ql_flag_can_be_turned_off() -> None:
    """--flag is on by default and --no-flag turns it off."""
    for args, expected in (([], True), (["--flag"], True), (["--no-flag"], False)):
        ctx = cli.process_ql.make_context("process-ql", args, resilient_parsing=True)
        assert ctx.params["flag"] is expected
