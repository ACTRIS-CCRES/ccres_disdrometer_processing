"""Tests the physics corrections of Zdd and Zdcr.

1. refractive index of water from Ellison (2007) at the radar frequency ;
2. |K|^2 of the radar band ;
3. scattering at the radar's own frequency ;
4. two-way rain attenuation added to Zdcr ;
5. radar window aligned on the disdrometer telegrams and their fall time.
"""

import logging

import numpy as np
import pandas as pd
import pytest
import toml
import xarray as xr

from ccres_disdrometer_processing import dielectric, scattering, utils
from ccres_disdrometer_processing.open_disdro_netcdf import (
    read_parsivel_cloudnet_choice,
)
from ccres_disdrometer_processing.open_radar_netcdf import (
    aligned_reflectivity,
    fall_time_lag,
    read_radar_cloudnet,
    read_radar_frequency,
    window_median,
)
from ccres_disdrometer_processing.processing.preprocessed_file2processed import (
    extract_dcr_data,
)

AXR = "BeardChuang_PolynomialFit"


# 1-3 : water properties at the radar frequency ---------------------------------


def test_ellison_static_limit_and_known_index():
    # at 0 Hz, the permittivity is the static one of Ellison's Eq. (8)
    t = 10.0
    eps_s = 87.9144 - 0.404399 * t + 9.58726e-4 * t**2 - 1.32802e-6 * t**3
    assert dielectric.ellison_permittivity(0.0, t) == pytest.approx(complex(eps_s))
    # values used in the Juelich study (35.5 GHz) and for the MIRA-10 (9.6 GHz)
    n = dielectric.ellison_refractive_index(35.5e9, 10.0)
    assert (n.real, n.imag) == pytest.approx((4.6943, 2.6061), abs=1e-4)
    n = dielectric.ellison_refractive_index(9.6e9, 10.0)
    assert (n.real, n.imag) == pytest.approx((7.7609, 2.4126), abs=1e-4)


def test_table_entries_follow_ellison_and_the_band_conventions():
    table = dielectric._table()
    assert len(table) >= 8
    for f, w in table.items():
        n = dielectric.ellison_refractive_index(f, w.temperature)
        assert w.refractive_index == pytest.approx(n, abs=1e-6)
        assert w.kw_sqr == dielectric.kw_sqr_convention(f)
        assert w.temperature == dielectric.T_REF
    assert dielectric.kw_sqr_convention(9.6e9) == 0.93
    assert dielectric.kw_sqr_convention(35.5e9) == 0.88
    assert dielectric.kw_sqr_convention(94e9) == 0.74
    assert dielectric.kw_sqr_convention(95e9) == 0.74


@pytest.mark.parametrize(
    ("radar", "used"),
    [
        (93.99968719482422e9, 94.0e9),  # RPG-94, as written by Cloudnet
        (35.00175094604492e9, 35.0e9),  # RPG-35
        (35.459999084472656e9, 35.46e9),  # MIRA-35 of Juelich
        (9.600000381469727e9, 9.6e9),  # MIRA-10
        (95.0e9, 95.0e9),  # BASTA
    ],
)
def test_radar_frequency_uses_its_table_entry(radar, used):
    w = dielectric.water_dielectric(radar)
    assert w.from_table
    assert w.frequency == used


def test_frequency_absent_from_the_table_is_computed(caplog):
    with caplog.at_level(logging.WARNING):
        w = dielectric.water_dielectric(77.004e9)
    assert not w.from_table
    assert w.frequency == pytest.approx(77.0e9)
    assert w.refractive_index == dielectric.ellison_refractive_index(77.0e9)
    assert w.kw_sqr == 0.74
    assert "not in water_dielectric.toml" in caplog.text


@pytest.mark.parametrize("radar", [np.nan, 0.0, -1.0])
def test_invalid_radar_frequency_is_an_error(radar):
    with pytest.raises(ValueError, match="radar frequency"):
        dielectric.water_dielectric(radar)


def test_kw_sqr_scales_the_backscattering_only():
    D = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32)
    w = dielectric.water_dielectric(94e9)
    ref = scattering.scattering_prop(D, 1, w.frequency, e=w.refractive_index)
    new = scattering.scattering_prop(
        D, 1, w.frequency, e=w.refractive_index, kw_sqr=w.kw_sqr
    )
    for v in ("bscat_mie", "bscat_tmatrix"):
        np.testing.assert_allclose(getattr(new, v), getattr(ref, v) * 0.93 / 0.74)
    np.testing.assert_array_equal(new.att_tmatrix, ref.att_tmatrix)
    # the table itself is left untouched
    again = scattering.scattering_prop(D, 1, w.frequency, e=w.refractive_index)
    np.testing.assert_array_equal(again.bscat_tmatrix, ref.bscat_tmatrix)


def test_rayleigh_drops_follow_the_index_and_kw_change():
    D = np.array([0.1], dtype=np.float32)  # Rayleigh regime at 35 GHz
    old = scattering.scattering_prop(D, 1, 35e9, e=2.99645 + 1.54866j)
    w = dielectric.water_dielectric(35e9)
    new = scattering.scattering_prop(
        D, 1, w.frequency, e=w.refractive_index, kw_sqr=w.kw_sqr
    )

    def k2(m):
        e = m * m
        return abs((e - 1) / (e + 2)) ** 2

    expected = k2(w.refractive_index) / k2(2.99645 + 1.54866j) * 0.93 / 0.88
    assert new.bscat_tmatrix[0] / old.bscat_tmatrix[0] == pytest.approx(
        expected, rel=2e-3
    )


# 5 : radar aligned on the disdrometer telegrams ---------------------------------


def test_window_median_takes_the_samples_of_each_window():
    rt = np.arange(0.0, 300.0, 3.0)  # a profile every 3 s
    z = rt.copy()
    lo, hi = np.array([0.0, 57.0, 200.0]), np.array([60.0, 117.0, 260.0])
    out = window_median(rt, z, lo, hi, nmin=12)
    # (0, 60] holds 3, ..., 60 ; (57, 117] holds 60, ..., 117 ; (200, 260] 201..258
    np.testing.assert_allclose(out, [31.5, 88.5, 229.5])
    z[rt > 230] = np.nan
    assert np.isfinite(window_median(rt, z, lo, hi, nmin=12)[2])  # NaN ignored
    assert np.isnan(window_median(rt, z, lo, hi, nmin=25)[0])  # too few samples


def test_fall_time_lag():
    rng = np.array([100.0, 200.0, 300.0])
    dv = np.full((10, 3), -4.0)
    rain = np.zeros(10, bool)
    np.testing.assert_array_equal(fall_time_lag(rng, dv, rain), 0.0)
    rain[3:] = True
    np.testing.assert_allclose(fall_time_lag(rng, dv, rain, 0.5), 0.5 * rng / 4.0)
    # a gate without rain-like fall speed (melting layer...) is not used for |v|
    dv[:, 1] = -0.2
    np.testing.assert_allclose(fall_time_lag(rng, dv, rain, 0.5), 0.5 * rng / 4.0)


def ramp_radar(day="2024-12-06"):
    """Radar data whose Zh is the time (s since 00:00), every 3 s."""
    t = pd.date_range(f"{day} 00:00:00", f"{day} 23:59:57", freq="3s")
    secs = (t - t[0]).total_seconds().values
    rng = np.array([100.0, 200.0])
    return xr.Dataset(
        {"Zh": (("time", "range"), np.repeat(secs[:, None], rng.size, axis=1))},
        coords={"time": t, "range": rng},
    )


def test_alignment_window_follows_the_telegram_and_the_fall():
    radar = ramp_radar()
    labels = pd.date_range("2024-12-06", periods=1440, freq="1min")
    telegram = (labels + pd.Timedelta(seconds=56)).values.copy()
    telegram[5] = np.datetime64("NaT")
    rain = np.ones(1440, bool)
    dv = np.full((1440, 2), -5.0)
    z, lag = aligned_reflectivity(radar, labels, telegram, rain, dv, factor=0.5)
    np.testing.assert_allclose(lag, [10.0, 20.0])
    # minute 10 : telegram at 10:56 marks the end of (9:56, 10:56] ; the radar
    # window is advanced by the lag : median of the samples of (s - 60 - lag,
    # s - lag]
    for j in range(2):
        s = 10 * 60 + 56 - lag[j]
        samples = np.arange(0.0, 86400.0, 3.0)
        expected = np.median(samples[(samples > s - 60) & (samples <= s)])
        assert z[10, j] == expected
    # without telegram : the minute around its label, advanced the same way
    samples = np.arange(0.0, 86400.0, 3.0)
    s = 5 * 60 - lag[0]
    expected = np.median(samples[(samples > s - 30) & (samples <= s + 30)])
    assert z[5, 0] == expected
    # "start" convention : the telegram time marks the beginning of its interval
    z_start, _ = aligned_reflectivity(
        radar, labels, telegram, rain, dv, factor=0.5, convention="start"
    )
    np.testing.assert_allclose(z_start[10] - z[10], 60.0)
    with pytest.raises(ValueError, match="convention"):
        aligned_reflectivity(radar, labels, telegram, rain, dv, convention="middle")


def write_radar_file(path, day="2024-12-06"):
    times = pd.date_range(f"{day} 00:00:15", f"{day} 23:59:45", freq="3s")
    ranges = np.array([100.0, 200.0, 300.0])
    shape = (times.size, ranges.size)
    ds = xr.Dataset(
        {
            "Zh": (("time", "range"), np.full(shape, 10.0), {"units": "dBZ"}),
            "v": (("time", "range"), np.full(shape, -4.0), {"units": "m s-1"}),
            "radar_frequency": ((), 93.99968719482422, {"units": "GHz"}),
            "latitude": ((), 46.8, {"units": "degree_north"}),
            "longitude": ((), 6.9, {"units": "degree_east"}),
            "altitude": ((), 491.0, {"units": "m"}),
        },
        coords={"time": times, "range": ("range", ranges, {"units": "m"})},
        attrs={"source": "RPG-Radiometer Physics RPG-FMCW-94", "instrument_pid": "pid"},
    )
    ds.to_netcdf(path)


def test_radar_reader_aligns_on_the_disdrometer(tmp_path):
    path = tmp_path / "radar.nc"
    write_radar_file(path)
    assert read_radar_frequency(path) == pytest.approx(93.99968719482422e9)
    labels = pd.date_range("2024-12-06", periods=1440, freq="1min")
    disdro = xr.Dataset(
        {
            "disdro_telegram_time": ("time", (labels + pd.Timedelta("56s")).values),
            "disdro_pr": ("time", np.r_[np.zeros(600), np.ones(840)]),
        },
        coords={"time": labels},
    )
    plain = read_radar_cloudnet(path)
    assert "Zdcr_aligned" not in plain
    radar = read_radar_cloudnet(path, disdro=disdro, fall_time_factor=0.55)
    np.testing.assert_array_equal(radar.Zdcr.values, plain.Zdcr.values)
    np.testing.assert_allclose(
        radar.fall_time_lag.values, 0.55 * radar.range.values / 4.0, rtol=1e-6
    )
    # constant Zh : every complete window gives it
    assert np.all(radar.Zdcr_aligned.values[5:-5] == 10.0)


def test_preprocessed_telegram_time(data_input_dir, data_conf_dir):
    source = utils.get_file_from_cloudnet(
        "juelich",
        "2024-02-08",
        "parsivel",
        "https://hdl.handle.net/21.12132/3.2a1ca46ed70c4929",
        data_input_dir,
    )
    config = toml.load(data_conf_dir / "config_juelich_mira-parsivel.toml")
    data = read_parsivel_cloudnet_choice(source, [35e9], config)
    with xr.open_dataset(source) as raw:
        written = raw.time.values
    t = data.disdro_telegram_time.values
    kept = ~np.isnat(t)
    assert kept.sum() > 1000
    assert np.isin(t[kept], written).all()
    # each telegram lies in the minute (t - 30 s, t + 30 s] of its label
    offset = (t[kept] - data.time.values[kept]) / np.timedelta64(1, "s")
    assert np.all((offset > -30) & (offset <= 30))


# 4 : rain attenuation ------------------------------------------------------------


def test_zdcr_gets_the_aligned_and_attenuation_corrected_value():
    t = pd.date_range("2024-12-06", periods=4, freq="1min")
    rng = np.array([150.0, 300.0])
    ds = xr.Dataset(
        {
            "Zdcr": (("time", "range"), np.full((4, 2), 20.0)),
            "Zdcr_aligned": (("time", "range"), np.full((4, 2), 21.0)),
            "DVdcr": (("time", "range"), np.full((4, 2), -4.0)),
            "Zdlog_vfov_modv_tm": (
                ("time", "radar_frequencies"),
                np.full((4, 1), 19.0),
            ),
            "attd_vfov_modv": (
                ("time", "radar_frequencies"),
                np.array([[0.0], [1.0], [2.0], [np.nan]]),
            ),
            "radar_frequency": ("time", np.full(4, 93.9997e9)),
            "psd": (("time", "size", "speed"), np.ones((4, 2, 2))),
            "measV": (("time", "size"), np.full((4, 2), 4.0)),
        },
        coords={"time": t, "range": rng, "radar_frequencies": [94e9]},
    )
    conf = {"instrument_parameters": {"PROCESSING_RANGES": [150.0, 300.0]}}
    out = extract_dcr_data(ds, conf)
    att = 2 * np.array([0.0, 1.0, 2.0, 0.0])[:, None] * rng[None, :] / 1000
    np.testing.assert_allclose(out.dcr_attenuation.values, att, rtol=1e-6)
    np.testing.assert_allclose(out.Zdcr.values, 21.0 + att, rtol=1e-6)
    np.testing.assert_allclose(out.Delta_Z.values, 2.0 + att, rtol=1e-6)
    # a preprocessed file without the alignment uses its 1-minute Zdcr
    out = extract_dcr_data(ds.drop_vars("Zdcr_aligned"), conf)
    np.testing.assert_allclose(out.Zdcr.values, 20.0 + att, rtol=1e-6)


def test_alignment_accepts_radars_sending_profiles_in_bursts():
    # 4 profiles 3 s apart at the start of each minute : median gap 3 s, but
    # only 4 profiles per window, all of them to be used
    day = pd.Timestamp("2024-12-06")
    t = pd.DatetimeIndex(
        [
            day + pd.Timedelta(minutes=m, seconds=s)
            for m in range(1440)
            for s in (0, 3, 6, 9)
        ]
    )
    radar = xr.Dataset(
        {"Zh": (("time", "range"), np.full((t.size, 1), 10.0))},
        coords={"time": t, "range": [100.0]},
    )
    labels = pd.date_range(day, periods=1440, freq="1min")
    telegram = (labels + pd.Timedelta(seconds=56)).values
    z, _ = aligned_reflectivity(
        radar, labels, telegram, np.zeros(1440, bool), np.full((1440, 1), -5.0)
    )
    assert np.isfinite(z[5:-5]).all()
