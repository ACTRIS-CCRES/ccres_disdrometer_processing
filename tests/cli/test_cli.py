"""Tests for `ccres_disdrometer_processing` package."""

import toml
import xarray as xr
from click.testing import CliRunner

from ccres_disdrometer_processing.cli import cli
from ccres_disdrometer_processing.plot import plot, utils
from tests.station_conf import station_conf


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


def test_process_ql_reads_min_points_in_the_configuration(tmp_path, monkeypatch):
    """The minimum number of good timesteps comes from [thresholds] MIN_POINTS."""
    conf = tmp_path / "conf.toml"
    conf.write_text(toml.dumps(station_conf(thresholds={"MIN_POINTS": 17})))
    files = {name: tmp_path / f"{name}.nc" for name in ("processed", "preprocessed")}
    for f in files.values():
        f.touch()
    used = []
    monkeypatch.setattr(utils, "read_nc", lambda f: xr.Dataset({"events": ("n", [0])}))
    monkeypatch.setattr(utils, "read_and_concatenante_preprocessed_ds", lambda *a: 0)
    for name in ("plot_processed_ql_summary", "plot_processed_ql_detailled"):
        monkeypatch.setattr(plot, name, lambda *a: used.append(a[-1]))
    args = ["--process-today", files["processed"]]
    args += ["--preprocess-today", files["preprocessed"], "--config-file", conf]
    args += ["--prefix-output-ql-summary", str(tmp_path / "summary")]
    args += ["--prefix-output-ql-detailled", str(tmp_path / "detailled")]
    result = CliRunner().invoke(cli.process_ql, [str(a) for a in args])
    assert result.exit_code == 0, result.output
    assert used == [17, 17]
