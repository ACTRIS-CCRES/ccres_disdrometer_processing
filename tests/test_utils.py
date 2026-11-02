"""Tests for the helpers in ccres_disdrometer_processing.utils."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ccres_disdrometer_processing.utils import (
    compressed_encoding,
    format_ql_file_prefix,
    instrument_position,
    scalar_positions,
)


def test_format_ql_file_prefix_adds_the_index_pattern() -> None:
    assert format_ql_file_prefix("ql_summary") == "ql_summary_{:02d}.png"


def test_format_ql_file_prefix_drops_the_extension() -> None:
    assert format_ql_file_prefix("ql_summary.png") == "ql_summary_{:02d}.png"
    assert format_ql_file_prefix("out/ql.png") == "out/ql_{:02d}.png"


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (np.float32(45.1), 45.1),  # older Cloudnet files : scalar
        (np.array([np.nan, 45.1, 45.1], dtype="float32"), 45.1),  # first is NaN
        (np.array([np.nan, np.nan]), np.nan),
    ],
)
def test_instrument_position(values, expected) -> None:
    dims = () if np.ndim(values) == 0 else ("time",)
    position = xr.DataArray(values, dims=dims, attrs={"units": "degree_north"})
    result = instrument_position(position)
    assert result.ndim == 0
    assert result.attrs == {"units": "degree_north"}
    np.testing.assert_allclose(result.values, expected, rtol=1e-6)


def test_scalar_positions_drops_time_bins() -> None:
    # layout of files preprocessed before positions were made scalar
    time = pd.date_range("2025-01-01", periods=3, freq="1min")
    ds = xr.Dataset(
        {
            "disdro_latitude": ("time_bins", np.full(3, 45.1)),
            "ams_latitude": ("time", np.full(3, 45.2)),
            "disdro_altitude": ((), 10.0),
            "Zdd": ("time", np.arange(3.0)),
        },
        coords={"time": time, "time_bins": time},
    )
    result = scalar_positions(ds)
    assert "time_bins" not in result.dims
    assert result["disdro_latitude"].ndim == 0
    assert result["ams_latitude"].ndim == 0
    assert float(result["disdro_altitude"]) == 10.0
    assert result["Zdd"].dims == ("time",)


def test_compressed_encoding_keeps_existing_settings() -> None:
    ds = xr.Dataset(
        {
            "psd": (("time", "size"), np.zeros((2, 3))),
            "flag": ("time", np.array([True, False])),
            "F": ((), 0.0054),
            "model": ("time", np.array(["a", "b"], dtype=object)),
        },
        coords={"time": pd.date_range("2024-01-01", periods=2, freq="1min")},
    )
    enc = compressed_encoding(ds, {"psd": {"_FillValue": -9.0}, "time": {"units": "s"}})
    assert enc["psd"] == {"_FillValue": -9.0, "zlib": True, "complevel": 4}
    assert enc["flag"]["zlib"]
    assert "F" not in enc and "model" not in enc  # scalar and strings left as they are
    assert enc["time"] == {"units": "s"}
