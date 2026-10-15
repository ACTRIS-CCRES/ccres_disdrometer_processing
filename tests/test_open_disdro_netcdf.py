"""Tests the Parsivel2 data_raw axis order across cloudnetpy versions."""

import numpy as np
import pytest
import toml
import xarray as xr

from ccres_disdrometer_processing import utils
from ccres_disdrometer_processing.open_disdro_netcdf import (
    read_parsivel_cloudnet_choice,
)

RADAR_FREQUENCIES = [35.0e9, 94.0e9]


def write_layout(ds, raw, version, path):
    """Write ds with data_raw replaced by raw and cloudnetpy_version by version."""
    out = ds.copy()
    out["data_raw"] = (("time", "diameter", "velocity"), raw, ds["data_raw"].attrs)
    out.attrs["cloudnetpy_version"] = version
    out.to_netcdf(path)
    return path


@pytest.fixture
def parsivel_files(data_input_dir, data_conf_dir, tmp_path):
    """Build the same Parsivel2 day in its cloudnetpy < 1.94 and >= 1.94 layouts.

    cloudnetpy < 1.94 stored data_raw as (time, velocity, diameter) despite its
    (time, diameter, velocity) dimension names; 1.94 fixed the order. Cloudnet
    reprocesses old days, so the downloaded file may come in either layout:
    both are rebuilt from it rather than relying on the version served.
    """
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
    major, minor = ds.attrs["cloudnetpy_version"].split(".")[:2]
    raw = ds["data_raw"].values  # (time, diameter, velocity) from 1.94 on
    if (int(major), int(minor)) < (1, 94):
        raw = np.swapaxes(raw, 1, 2)

    old_file = write_layout(
        ds, np.swapaxes(raw, 1, 2), "1.93.0", tmp_path / "parsivel_1.93.nc"
    )
    new_file = write_layout(ds, raw, "1.94.0", tmp_path / "parsivel_1.94.nc")
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
