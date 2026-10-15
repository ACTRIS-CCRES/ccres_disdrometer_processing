"""Tests the Parsivel2 data_raw axis order across cloudnetpy versions."""

import numpy as np
import pytest
import toml
import xarray as xr

from ccres_disdrometer_processing.open_disdro_netcdf import (
    read_parsivel_cloudnet_choice,
)

RADAR_FREQUENCIES = [35.0e9, 94.0e9]


@pytest.fixture
def parsivel_files(data_input_dir, data_conf_dir, tmp_path):
    """Build a cloudnetpy < 1.94 Parsivel2 file and its cloudnetpy >= 1.94 equivalent.

    cloudnetpy < 1.94 stored data_raw as (time, velocity, diameter) despite its
    (time, diameter, velocity) dimension names; 1.94 fixed the order. The
    second file is the first one rewritten the way cloudnetpy >= 1.94 does.
    """
    old_file = data_input_dir / "20240208_juelich_parsivel_2a1ca46e.nc"
    config = toml.load(data_conf_dir / "config_juelich_mira-parsivel.toml")

    with xr.open_dataset(old_file) as ds:
        assert ds.attrs["cloudnetpy_version"].startswith("1.58.")
        new = ds.load().copy()
    new["data_raw"] = (
        ("time", "diameter", "velocity"),
        np.swapaxes(new["data_raw"].values, 1, 2),
        new["data_raw"].attrs,
    )
    new.attrs["cloudnetpy_version"] = "1.94.0"
    new_file = tmp_path / "20240208_juelich_parsivel_2a1ca46e_1.94.nc"
    new.to_netcdf(new_file)

    return old_file, new_file, config


def mean_fallspeed(psd, size_classes, speed_classes, d_min, d_max):
    """Mean fall speed (m/s) of the drops between d_min and d_max (mm)."""
    sel = (size_classes > d_min) & (size_classes < d_max)
    counts = np.nansum(psd, axis=0)[sel]
    return np.nansum(counts * speed_classes[None, :]) / np.nansum(counts)


def test_psd_axis_order_follows_cloudnetpy_version(parsivel_files):
    old_file, new_file, config = parsivel_files

    old = read_parsivel_cloudnet_choice(old_file, RADAR_FREQUENCIES, config)
    new = read_parsivel_cloudnet_choice(new_file, RADAR_FREQUENCIES, config)

    assert old.psd.dims == ("time", "size_classes", "speed_classes")
    np.testing.assert_array_equal(old.psd.values, new.psd.values)

    # 1-2 mm raindrops fall at ~4-6 m/s (Gunn and Kinzer); swapped axes give
    # well under 1 m/s.
    v = mean_fallspeed(
        old.psd.values, old.size_classes.values, old.speed_classes.values, 1, 2
    )
    assert 3 < v < 7
