"""Tests for the fall speed / diameter fit used in the quicklooks."""

import numpy as np
import xarray as xr

from ccres_disdrometer_processing.plot.utils import get_y_fit_dd


def make_psd_dataset(psd: np.ndarray) -> xr.Dataset:
    """Build a minimal disdrometer dataset holding only a psd (time, size, speed)."""
    return xr.Dataset(
        {"psd": (("time", "size_classes", "speed_classes"), psd)},
        coords={
            "size_classes": np.linspace(0.2, 5, 4),
            "speed_classes": np.linspace(0.5, 10, 4),
        },
    )


def test_get_y_fit_dd_without_drops_returns_no_data_flag() -> None:
    result = get_y_fit_dd(make_psd_dataset(np.zeros((2, 4, 4))))
    assert result == (0, 0, 0, 0, 0, 0)


def test_get_y_fit_dd_with_too_few_drops_returns_no_fit_flag() -> None:
    # one drop gives one point for a three-parameter fit, so curve_fit fails
    psd = np.zeros((2, 4, 4))
    psd[0, 1, 2] = 1
    result = get_y_fit_dd(make_psd_dataset(psd))
    assert result == (-1, -1, -1, -1, -1, -1)
