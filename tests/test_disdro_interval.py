"""Tests the normalisation of the disdrometer counts by their measurement interval."""

import numpy as np
import pytest
import toml
import xarray as xr

from ccres_disdrometer_processing import scattering, utils
from ccres_disdrometer_processing.open_disdro_netcdf import (
    read_parsivel_cloudnet_choice,
    reflectivity_model_multilambda_measmodV_hvfov,
)

FREQUENCY = 35.0e9


@pytest.fixture
def juelich_day(data_input_dir, data_conf_dir):
    source = utils.get_file_from_cloudnet(
        "juelich",
        "2024-02-08",
        "parsivel",
        "https://hdl.handle.net/21.12132/3.2a1ca46ed70c4929",
        data_input_dir,
    )
    config = toml.load(data_conf_dir / "config_juelich_mira-parsivel.toml")
    with xr.open_dataset(source) as ds:
        ds = ds.load()
    return ds, config


def with_interval(ds, interval, path):
    """Write ds with every telegram's interval set to interval (None drops it)."""
    out = ds.copy()
    if interval is None:
        out = out.drop_vars("interval")
    else:
        out["interval"] = ("time", np.broadcast_to(interval, ds.time.shape).copy())
    out.to_netcdf(path)
    return path


def zdd(path, config):
    data = read_parsivel_cloudnet_choice(path, [FREQUENCY], config)
    scatt = [
        scattering.scattering_prop(data.size_classes[0:-5], fov, FREQUENCY)
        for fov in (1, 0)
    ]
    out = reflectivity_model_multilambda_measmodV_hvfov(
        data, scatt, len(data.size_classes[0:-5])
    )
    return out, out["Zdlog_vfov_modv_tm"].isel(radar_frequencies=0).values


def test_counts_are_normalised_by_their_interval(juelich_day, tmp_path):
    ds, config = juelich_day
    nominal, z60 = zdd(with_interval(ds, 60.0, tmp_path / "i60.nc"), config)
    missing, z_none = zdd(with_interval(ds, None, tmp_path / "none.nc"), config)
    short, z5 = zdd(with_interval(ds, 5.0, tmp_path / "i5.nc"), config)
    long, z63 = zdd(with_interval(ds, 63.0, tmp_path / "i63.nc"), config)

    rain = np.isfinite(z60)
    assert rain.sum() > 100
    # no interval in the file: the nominal minute, as before
    np.testing.assert_allclose(z_none[rain], z60[rain])
    np.testing.assert_allclose(z5[rain] - z60[rain], 10 * np.log10(12), atol=1e-4)
    np.testing.assert_allclose(z63[rain] - z60[rain], 10 * np.log10(60 / 63), atol=1e-4)
    np.testing.assert_allclose(
        short.disdro_pr_from_raw.values[rain], 12 * nominal.disdro_pr_from_raw[rain]
    )

    assert (short.QF_disdro_interval.values[rain] == 0).all()
    assert (long.QF_disdro_interval.values[rain] == 1).all()
    # rain accumulation: a short telegram stands for the minute, a long one for
    # its own interval
    np.testing.assert_allclose(short.disdro_cp[-1], nominal.disdro_cp[-1])
    np.testing.assert_allclose(long.disdro_cp[-1], nominal.disdro_cp[-1] * 63 / 60)
    assert missing.disdro_cp[-1] == pytest.approx(float(nominal.disdro_cp[-1]))


def test_telegram_after_a_gap_is_dropped(juelich_day, tmp_path):
    ds, config = juelich_day
    data = read_parsivel_cloudnet_choice(
        with_interval(ds, 60.0, tmp_path / "i60.nc"), [FREQUENCY], config
    )
    wet = int(np.nanargmax(data.disdro_pr.values))
    interval = np.full(ds.time.size, 60.0)
    # the telegram kept for that minute reports the time since the previous one
    kept = int(np.argmin(np.abs(ds.time.values - data.time.values[wet])))
    interval[kept] = 30050.0
    gap = read_parsivel_cloudnet_choice(
        with_interval(ds, interval, tmp_path / "gap.nc"), [FREQUENCY], config
    )

    assert np.isnan(gap.disdro_interval.values[wet])
    assert gap.QF_disdro_interval.values[wet] == -9
    assert np.isnan(gap.psd.values[wet]).all()
    assert np.isnan(gap.disdro_pr.values[wet])
    others = np.arange(data.time.size) != wet
    np.testing.assert_array_equal(
        gap.disdro_interval.values[others], data.disdro_interval.values[others]
    )


def test_interval_given_once_for_the_file(juelich_day, tmp_path):
    ds, config = juelich_day
    per_telegram = read_parsivel_cloudnet_choice(
        with_interval(ds, 5.0, tmp_path / "i5.nc"), [FREQUENCY], config
    )
    once = ds.copy()
    once["interval"] = 5.0  # no time dimension, as in Cabauw files
    once.to_netcdf(tmp_path / "once.nc")
    scalar = read_parsivel_cloudnet_choice(tmp_path / "once.nc", [FREQUENCY], config)
    np.testing.assert_array_equal(
        scalar.disdro_interval.values, per_telegram.disdro_interval.values
    )


def test_disdro_minutes():
    ds = xr.Dataset(
        {"disdro_interval": ("time", [60.0, 63.0, 5.0, np.nan])},
        coords={"time": np.arange(4)},
    )
    np.testing.assert_allclose(utils.disdro_minutes(ds), [1, 1.05, 1, 1])
    np.testing.assert_array_equal(
        utils.disdro_minutes(ds.drop_vars("disdro_interval")), np.ones(4)
    )
