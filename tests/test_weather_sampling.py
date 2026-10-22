"""Tests the handling of weather stations sampling less often than every minute."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ccres_disdrometer_processing.processing import (
    preprocessed_file2processed_weather as weather,
)

CONF = {
    "thresholds": {
        "MIN_RAINFALL_AMOUNT": 3,
        "MAX_RR": 50,
        "DD_FALLSPEED_RATIO": 0.3,
        "MIN_TEMP": 2,
        "MAX_WS": 7,
        "DD_ANGLE": 45,
        "MIN_HUR": 0,
        "MAX_HUR": 100,
        "DD_RG_MAX_PR_ACC_RATIO": 0.3,
    },
    "instrument_parameters": {"DD_ORIENTATION": 0},
}


def make_ds(sampling_mn, n_hours=6, missing=()):
    """Build a preprocessed-like dataset on a 1-mn grid.

    AMS variables are only set every sampling_mn minutes (NaN elsewhere), as
    done by the preprocessing ; the sample indices in missing are dropped.
    """
    time = pd.date_range("2024-01-01", periods=n_hours * 60, freq="1min")
    sampled = np.zeros(time.size, dtype=bool)
    sampled[::sampling_mn] = True
    for k in missing:
        sampled[k * sampling_mn] = False

    def ams(value):
        return ("time", np.where(sampled, value, np.nan))

    n_diam, n_speed = 2, 2
    return xr.Dataset(
        {
            "ta": ams(10.0),
            "ws": ams(3.0),
            "wd": ams(90.0),
            "hur": ams(80.0),
            "ams_pr": ams(6.0),  # mm/h
            "disdro_pr": ("time", np.full(time.size, 6.0)),
            "psd": (
                ("time", "diameter", "speed"),
                np.ones((time.size, n_diam, n_speed)),
            ),
            "modV": (("time", "diameter"), np.ones((time.size, n_diam))),
            "speed_classes": ("speed", np.ones(n_speed)),
        },
        coords={"time": time},
    )


@pytest.mark.parametrize(
    ("sampling", "missing", "expected"),
    [
        (1, (), 1.0),
        (10, (), 10.0),
        # a missing sample right at the start used to double the sampling
        (10, (1,), 10.0),
        (1, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 1.0),
    ],
)
def test_get_ams_time_sampling(sampling, missing, expected):
    ds = make_ds(sampling, missing=missing)
    assert weather.get_ams_time_sampling(ds) == expected


def test_get_ams_time_sampling_without_data():
    ds = make_ds(10)
    ds["ta"][:] = np.nan
    ds["ams_pr"][:] = np.nan
    assert weather.get_ams_time_sampling(ds) == 1.0


def test_fill_weather_to_1min_nearest():
    ds = make_ds(10)
    ds["ta"][:] = np.where(np.isfinite(ds["ta"]), np.arange(ds.time.size), np.nan)
    filled = weather.fill_weather_to_1min(ds, 10)
    # 00:04 -> 00:00 sample, 00:06 -> 00:10 sample
    assert filled["ta"].values[4] == 0
    assert filled["ta"].values[6] == 10
    for key in weather.WEATHER_QC_VARS:
        assert np.isfinite(filled[key].values).all()
    # the input dataset is left untouched
    assert np.isnan(ds["ta"].values[4])


def test_fill_weather_to_1min_keeps_long_gaps():
    # samples 00:30 to 01:00 missing : nearest samples are 00:20 and 01:10, so
    # 00:30 and 01:00 are filled (10 mn away) but 00:45 stays NaN (25 mn away)
    ds = make_ds(10, missing=(3, 4, 5, 6))
    filled = weather.fill_weather_to_1min(ds, 10)
    ta = filled["ta"].values
    assert np.isfinite(ta[30]) and np.isfinite(ta[70])
    assert np.isnan(ta[45])


@pytest.mark.parametrize("sampling", [1, 10])
def test_qc_accumulation_and_fill(sampling):
    ds = make_ds(sampling)
    start, end = [ds.time.values[0]], [ds.time.values[179]]
    qc = weather.compute_quality_checks_weather(ds, CONF, start, end)
    # 6 mm/h during 3 h -> 18 mm, whatever the AMS sampling
    n_samples = 180 // sampling
    expected = n_samples * 6.0 * sampling / 60
    assert qc["ams_cp_since_event_begin"].values[179] == pytest.approx(expected)
    assert expected == pytest.approx(18.0)
    # weather QCs are defined on every minute of the event
    event = qc["flag_event"].values.astype(bool)
    for key in ["QC_ta", "QC_ws", "QC_hur"]:
        assert qc[key].values[event].all()
