"""Tests that an event in progress at the start of the 3-day window keeps its begin."""

import numpy as np
import pandas as pd
import xarray as xr

from ccres_disdrometer_processing.processing.preprocessed_file2processed import (
    add_carried_accumulation,
    carry_over_events,
    event_begin,
)

CONF = {
    "thresholds": {
        "MAX_INTERVAL": 60,
        "MIN_RAINFALL_AMOUNT": 3,
        "DD_RG_MAX_PR_ACC_RATIO": 0.3,
    }
}


def yesterday_file(path, flag, begin, cumul):
    time = pd.date_range("2025-03-08", periods=1440, freq="1min")
    xr.Dataset(
        {
            "flag_event": (("time",), np.full(1440, flag, dtype="i2")),
            "event_begin": (("time",), np.full(1440, np.datetime64(begin, "ns"))),
            "disdro_cp_since_event_begin": (("time",), np.full(1440, cumul)),
        },
        coords={"time": time},
    ).to_netcdf(path)
    return path


def window():
    time = pd.date_range("2025-03-08", periods=3 * 1440, freq="1min")
    return xr.Dataset(coords={"time": time})


def test_event_at_the_window_start_takes_its_begin_from_the_day_before(tmp_path):
    f = yesterday_file(tmp_path / "j-1.nc", 1, "2025-03-07T15:45", 7.0)
    start = [np.asarray(np.datetime64("2025-03-08T00:01"))]
    new_start, carried = carry_over_events(window(), CONF, start, f)
    assert np.datetime64(new_start[0]) == np.datetime64("2025-03-07T15:45")
    assert carried == {
        0: (np.datetime64("2025-03-07T15:45"), np.datetime64("2025-03-08T00:01"))
    }


def test_event_starting_later_or_without_event_the_day_before_is_kept(tmp_path):
    f = yesterday_file(tmp_path / "j-1.nc", 1, "2025-03-07T15:45", 7.0)
    late = [np.asarray(np.datetime64("2025-03-08T03:00"))]
    assert carry_over_events(window(), CONF, late, f)[1] == {}
    g = yesterday_file(tmp_path / "dry.nc", 0, "NaT", 0.0)
    start = [np.asarray(np.datetime64("2025-03-08T00:01"))]
    assert carry_over_events(window(), CONF, start, g)[1] == {}
    assert carry_over_events(window(), CONF, start, None)[1] == {}


def test_carried_accumulation_continues_the_day_before(tmp_path):
    f = yesterday_file(tmp_path / "j-1.nc", 1, "2025-03-07T15:45", 7.0)
    time = window().time
    cp = np.zeros(time.size)
    cp[1:600] = np.linspace(0.1, 2.0, 599)  # 2 mm in the window, under 3 mm
    qc = xr.Dataset(
        {
            "disdro_cp_since_event_begin": (("time",), cp),
            "QF_rainfall_amount": (("time",), np.zeros(time.size, dtype="i2")),
        },
        coords={"time": time},
    )
    begin = np.datetime64("2025-03-07T15:45")
    s0, e = np.datetime64("2025-03-08T00:01"), np.datetime64("2025-03-08T09:59")
    out = add_carried_accumulation(qc, CONF, {0: (begin, s0)}, f, [e])
    # 7 mm before the window : the event passes the 3 mm threshold from its start
    assert np.isclose(out["disdro_cp_since_event_begin"].sel(time=s0), 7.0)
    assert np.isclose(out["disdro_cp_since_event_begin"].sel(time=e), 7.0 + 1.9)
    assert (out["QF_rainfall_amount"].sel(time=slice(s0, e)) == 1).all()
    assert (out["QF_rainfall_amount"].sel(time="2025-03-09T12:00") == 0).all()


def test_event_begin_marks_the_timesteps_of_each_event():
    qc = window()
    start = [np.asarray(np.datetime64("2025-03-07T15:45"))]
    end = [np.asarray(np.datetime64("2025-03-08T09:01"))]
    qc["event_begin"] = event_begin(qc, start, end)
    b = qc["event_begin"]
    assert b.sel(time="2025-03-08T09:01").values == np.datetime64("2025-03-07T15:45")
    assert np.isnat(b.sel(time="2025-03-08T09:02").values)


def test_event_begin_without_event_can_be_written(tmp_path):
    qc = window()
    qc["event_begin"] = event_begin(qc, [], [])
    qc.to_netcdf(
        tmp_path / "out.nc",
        encoding={
            "event_begin": {
                "units": "seconds since 2000-01-01T00:00:00.0Z",
                "calendar": "proleptic_gregorian",
            }
        },
    )
    with xr.open_dataset(tmp_path / "out.nc") as out:
        assert np.isnat(out["event_begin"].values).all()


def test_day_before_processed_without_event_begin_carries_nothing(tmp_path):
    # a file processed before event_begin existed
    path = yesterday_file(tmp_path / "old.nc", 1, "2025-03-07T15:45", 7.0)
    with xr.open_dataset(path) as ds:
        ds.drop_vars("event_begin").load().to_netcdf(tmp_path / "old_nobegin.nc")
    start = [np.asarray(np.datetime64("2025-03-08T00:01"))]
    new_start, carried = carry_over_events(
        window(), CONF, start, tmp_path / "old_nobegin.nc"
    )
    assert carried == {} and new_start == start


def test_event_beginning_at_the_window_start_is_not_carried(tmp_path):
    # in progress the day before, but it began at the first time of the window
    f = yesterday_file(tmp_path / "j-1.nc", 1, "2025-03-08T00:01", 0.5)
    start = [np.asarray(np.datetime64("2025-03-08T00:01"))]
    assert carry_over_events(window(), CONF, start, f)[1] == {}


def test_carried_event_checks_the_rain_gauge_on_the_whole_event(tmp_path):
    time = pd.date_range("2025-03-08", periods=1440, freq="1min")
    f = tmp_path / "j-1.nc"
    xr.Dataset(
        {
            "disdro_cp_since_event_begin": (("time",), np.full(1440, 3.0)),
            "ams_cp_since_event_begin": (("time",), np.full(1440, 10.0)),
        },
        coords={"time": time},
    ).to_netcdf(f)
    time = window().time
    cp = np.zeros(time.size)
    cp[1:600] = np.linspace(0.1, 2.0, 599)
    qc = xr.Dataset(
        {
            # disdrometer and rain gauge agree inside the window
            "disdro_cp_since_event_begin": (("time",), cp),
            "ams_cp_since_event_begin": (("time",), cp.copy()),
            "QF_rainfall_amount": (("time",), np.ones(time.size, dtype="i2")),
            "QF_rg_dd": (("time",), np.ones(time.size, dtype="i2")),
        },
        coords={"time": time},
    )
    begin = np.datetime64("2025-03-07T15:45")
    s0, e = np.datetime64("2025-03-08T00:01"), np.datetime64("2025-03-08T09:59")
    out = add_carried_accumulation(qc, CONF, {0: (begin, s0)}, f, [e])
    assert np.isclose(out["ams_cp_since_event_begin"].sel(time=e), 10.0 + 1.9)
    # 4.9 mm against 11.9 mm over the whole event: 59 % apart, above 30 %
    assert (out["QF_rg_dd"].sel(time=e) == 0).all()
    assert (out["QF_rg_dd"].sel(time="2025-03-09T12:00") == 1).all()
