"""Tests for the helpers in ccres_disdrometer_processing.utils."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ccres_disdrometer_processing.utils import (
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
