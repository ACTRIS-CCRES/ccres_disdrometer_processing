"""Tests the saturation detection and QC_sat on real MIRA-35 days."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from click.testing import CliRunner

from ccres_disdrometer_processing.cli import cli
from ccres_disdrometer_processing.open_radar_netcdf import read_radar_cloudnet

DATA = Path(__file__).parent / "data"
INPUTS = DATA / "inputs"
SATURATION_VARS = ["QF_saturation", "saturation_snr_cap", "Zdcr_saturation_ceiling"]


def input_file(name):
    path = INPUTS / name
    if not path.exists():
        pytest.skip(f"{name} not in the test inputs")
    return path


@pytest.mark.parametrize(
    ("name", "gate", "ceiling"),
    [
        # old MIRA of Lindenberg: 21.7 dBZ at 200 m, 18.7 dBZ near 300 m
        ("20240915_lindenberg_mira-35_d6cc3d73.nc", 201, 21.7),
        ("20240915_lindenberg_mira-35_d6cc3d73.nc", 288, 18.7),
        # Jülich: about 24 dBZ at 216 m
        ("20211204_juelich_mira.nc", 216, 24.0),
    ],
)
def test_heavy_rain_day_saturates_at_the_known_ceiling(name, gate, ceiling):
    radar = read_radar_cloudnet(input_file(name))
    at_gate = radar.sel(range=gate, method="nearest")
    assert abs(float(at_gate["Zdcr_saturation_ceiling"]) - ceiling) < 0.5
    assert (at_gate["QF_saturation"] == 1).sum() >= 20


@pytest.mark.parametrize(
    "name",
    ["20240917_lindenberg_mira-35_d6cc3d73.nc", "20211203_juelich_mira.nc"],
)
def test_day_without_heavy_rain_does_not_saturate(name):
    radar = read_radar_cloudnet(input_file(name))
    assert np.isnan(radar["Zdcr_saturation_ceiling"]).all()
    assert (radar["QF_saturation"] == 0).all()


def test_first_gate_of_juelich_does_not_saturate():
    # 144 m, the Delta Z gate of Jülich, stays under the ceiling
    radar = read_radar_cloudnet(input_file("20211204_juelich_mira.nc"))
    at_gate = radar.sel(range=144, method="nearest")
    assert np.isnan(float(at_gate["Zdcr_saturation_ceiling"]))
    assert (at_gate["QF_saturation"] == 0).all()


def run(command, args):
    result = CliRunner().invoke(command, [str(a) for a in args], catch_exceptions=False)
    assert result.exit_code == 0


@pytest.fixture(scope="module")
def lindenberg_20230922(tmp_path_factory):
    """Preprocess and process a saturated day, then the same day without the flag."""
    out = tmp_path_factory.mktemp("saturation")
    conf = DATA / "conf" / "config_lindenberg_mira-parsivel.toml"
    pre = out / "preprocessed.nc"
    run(
        cli.preprocess,
        [
            "--disdro-file",
            input_file("20230922_lindenberg_parsivel_1b0966f6.nc"),
            "--radar-file",
            input_file("20230922_lindenberg_mira_d6cc3d73.nc"),
            "--ws-file",
            input_file("20230922_lindenberg_weather-station_ffb25f43.nc"),
            "--config-file",
            conf,
            pre,
        ],
    )
    pro = out / "processed.nc"
    run(cli.process, ["--today", pre, "--config-file", conf, pro])
    # a file preprocessed before the saturation flag existed
    old_pre = out / "preprocessed_old.nc"
    with xr.open_dataset(pre) as ds:
        ds.drop_vars(SATURATION_VARS).to_netcdf(old_pre)
    old_pro = out / "processed_old.nc"
    run(cli.process, ["--today", old_pre, "--config-file", conf, old_pro])
    with (
        xr.open_dataset(pre) as p,
        xr.open_dataset(pro) as q,
        xr.open_dataset(old_pro) as o,
    ):
        yield p.load(), q.load(), o.load()


def test_preprocessing_writes_the_ceiling_and_both_flags(lindenberg_20230922):
    pre, _, _ = lindenberg_20230922
    at_gate = pre.sel(range=200, method="nearest")
    assert abs(float(at_gate["Zdcr_saturation_ceiling"]) - 21.7) < 0.5
    assert (at_gate["QF_saturation"] == 1).any()  # radar at the ceiling
    assert (at_gate["QF_saturation"] == 2).any()  # Zdd within 4 dB of it


def test_qc_saturation_follows_the_flag_and_enters_qc_overall(lindenberg_20230922):
    pre, pro, _ = lindenberg_20230922
    qf = pre["QF_saturation"].sel(range=200, method="nearest")
    qc = pro["QC_sat"]
    assert (qc == 0).sum() > 0
    assert ((qc == 1) == (qf.sel(time=qc.time) == 0)).all()
    assert (pro["QC_overall"].where(qc == 0) == 0).sum() == (qc == 0).sum()


def test_file_without_the_flag_keeps_its_qc_overall(lindenberg_20230922):
    _, pro, old = lindenberg_20230922
    assert (old["QC_sat"] == 1).all()
    assert "not assessed" in old["QC_sat"].attrs["comment"]
    # QC_sat only removes minutes: the other checks are unchanged
    kept = pro["QC_sat"] == 1
    assert (pro["QC_overall"].where(kept) == old["QC_overall"].where(kept)).sum() == (
        kept.sum()
    )
    assert (pro["QC_overall"] <= old["QC_overall"]).all()
