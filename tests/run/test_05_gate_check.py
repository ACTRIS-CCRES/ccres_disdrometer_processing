"""Tests gate_check on the Juelich files preprocessed and processed by the tests."""

import json

import pandas as pd
import pytest

from ccres_disdrometer_processing import gate_check
from tests.station_conf import CONF_FILE, station_conf

COMBO = "juelich_mira-35_0366fa69_parsivel_2a1ca46e"
DAYS = ["2021-12-02", "2021-12-03", "2021-12-04", "2021-12-05", "2021-12-06"]
DAYS += ["2024-02-08"]


@pytest.fixture
def daily_files(data_out_dir, tmp_path):
    """Link the test outputs under the daily names of the network processing."""
    directory = tmp_path / "files"
    directory.mkdir()
    for day in DAYS:
        for kind in ("preprocessed", "processed"):
            source = data_out_dir / f"juelich_{day}_mira-parsivel_{kind}.nc"
            link = directory / f"{COMBO}_{day.replace('-', '')}_{kind}.nc"
            link.symlink_to(source)
    return directory


def test_run(daily_files, tmp_path):
    """Run the command line, the periods cut by an instrument log entry."""
    log = tmp_path / "log.json"
    entry = {"instrument": "mira-35 0366fa69", "event": "maintenance"}
    entry["date"] = "2023-05-10"
    log.write_text(json.dumps({"entries": [entry]}))
    out = tmp_path / "gate"
    args = ["--combo", COMBO, "--config", str(CONF_FILE), "--period", "log"]
    args += ["--preprocessed-dir", str(daily_files), "--processed-dir"]
    args += [str(daily_files), "--instrument-log", str(log), "--output-dir", str(out)]
    gate_check.main(args)

    # only the 2024-02-08 event has MIN_POINTS QC-passed minutes
    events = pd.read_csv(out / f"{COMBO}_gate_events.csv")
    assert len(events) == 1
    assert events.start[0].startswith("2024-02-08")
    assert events.period[0] == "depuis 2023-05-10"
    summary = pd.read_csv(out / f"{COMBO}_gate_summary.csv")
    assert list(summary.set) == [
        "tous (toute la période)",
        "tous les événements",
        "depuis 2023-05-10",
    ]
    assert (summary.antenna == "monostatique").all()
    assert (summary.conf_gate == 150.0).all()
    overall = summary.iloc[0]
    assert abs(overall.dev_at_recommended) <= 0.25
    # the normalised reflectivity, on the same event, and its comparison
    assert overall.n_events_zn == 1
    # on this event Delta Z dips to -0.6 dB at 252-324 m, the normalised Z does not
    assert overall.gate_recommended == pytest.approx(359.8, abs=0.1)
    assert overall.gate_recommended_zn == pytest.approx(251.8, abs=0.1)
    assert overall.gate_shift == -3 and not overall.methods_agree
    assert overall.dev_at_zn_gate < -0.25
    assert abs(overall.dev_zn_at_dz_gate) <= 0.25
    assert events.zn_fit_ok[0]
    # a single event : too few for the period to be assessed
    assert summary.level.iloc[-1] == "-"
    assert (out / f"{COMBO}_gate.png").stat().st_size > 0


def test_no_good_event_is_an_error(daily_files):
    conf = station_conf(thresholds={"MIN_POINTS": 1000})
    with pytest.raises(ValueError, match="no good event"):
        gate_check.evaluate(COMBO, daily_files, daily_files, conf)
