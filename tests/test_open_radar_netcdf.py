"""Tests the reading and 1-mn resampling of Cloudnet radar files."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ccres_disdrometer_processing.open_radar_netcdf import read_radar_cloudnet


def write_radar_file(path, times):
    ranges = np.array([100.0, 200.0, 300.0])
    shape = (times.size, ranges.size)
    ds = xr.Dataset(
        {
            "Zh": (("time", "range"), np.full(shape, 10.0), {"units": "dBZ"}),
            "v": (("time", "range"), np.full(shape, -4.0), {"units": "m s-1"}),
            "radar_frequency": ((), 94.0, {"units": "GHz"}),
            "latitude": ((), 46.8, {"units": "degree_north"}),
            "longitude": ((), 6.9, {"units": "degree_east"}),
            "altitude": ((), 491.0, {"units": "m"}),
        },
        coords={"time": times, "range": ("range", ranges, {"units": "m"})},
        attrs={"source": "RPG-Radiometer Physics RPG-FMCW-94", "instrument_pid": "pid"},
    )
    ds.to_netcdf(path)


@pytest.mark.parametrize("sample_at_next_midnight", [False, True])
def test_time_grid_covers_the_file_day_only(tmp_path, sample_at_next_midnight):
    times = pd.date_range("2024-12-06 00:00:15", "2024-12-06 23:59:45", freq="30s")
    if sample_at_next_midnight:
        # some Cloudnet radar files end with a sample at 00:00 the next day
        times = times.append(pd.DatetimeIndex(["2024-12-07 00:00:00"]))
    path = tmp_path / "radar.nc"
    write_radar_file(path, times)

    radar = read_radar_cloudnet(path)

    assert radar.sizes["time"] == 1440
    assert radar.time.values[0] == np.datetime64("2024-12-06T00:00")
    assert radar.time.values[-1] == np.datetime64("2024-12-06T23:59")
    assert radar["Zdcr"].shape == (1440, 3)
    assert np.isfinite(radar["Zdcr"].values).all()
