"""Tests the detection of the radar receiver saturation and its quality check."""

import numpy as np
import pandas as pd
import xarray as xr

from ccres_disdrometer_processing import saturation
from ccres_disdrometer_processing.open_radar_netcdf import read_radar_cloudnet

RNG = np.random.default_rng(0)
N = 20000  # about one day of MIRA profiles


def heavy_rain_snr(n=N, cap=None):
    """SNR of a day with rain: a smooth tail, clamped at cap when given."""
    s = 40 + 8 * RNG.standard_normal(n)
    if cap is not None:
        s = np.where(s > cap, cap - 0.3 * RNG.random(n), s)
    return s


def test_clamped_gate_is_detected_and_smooth_gate_is_not():
    snr = np.column_stack([heavy_rain_snr(cap=55.0), heavy_rain_snr()])
    zh = snr - 40
    snr_cap, zh_ceiling, at_ceiling = saturation.detect_ceiling(snr, zh)
    assert abs(snr_cap[0] - 55.0) < 0.5
    assert np.isfinite(zh_ceiling[0])
    assert at_ceiling[:, 0].sum() >= saturation.N_STRONG
    assert np.isnan(snr_cap[1]) and np.isnan(zh_ceiling[1])
    assert not at_ceiling[:, 1].any()


def test_low_snr_pile_up_is_not_saturation():
    s = 20 + 5 * RNG.standard_normal(N)
    s = np.where(s > 25, 25 - 0.3 * RNG.random(N), s)  # pile-up, but at 25 dB
    snr_cap, _, _ = saturation.detect_ceiling(s[:, None], s[:, None])
    assert np.isnan(snr_cap[0])


def write_radar(path, with_snr):
    times = pd.date_range("2025-05-05 00:00:02", periods=N, freq="4s")
    ranges = np.array([200.0, 300.0])
    snr = np.column_stack([heavy_rain_snr(cap=55.0), heavy_rain_snr()])
    variables = {
        "Zh": (("time", "range"), snr - 40, {"units": "dBZ"}),
        "v": (("time", "range"), np.full(snr.shape, -5.0), {"units": "m s-1"}),
        "radar_frequency": ((), 35.5, {"units": "GHz"}),
        "latitude": ((), 48.1, {"units": "degree_north"}),
        "longitude": ((), 11.6, {"units": "degree_east"}),
        "altitude": ((), 520.0, {"units": "m"}),
    }
    if with_snr:
        variables["SNR"] = (("time", "range"), snr, {"units": "dB"})
    xr.Dataset(
        variables,
        coords={"time": times, "range": ("range", ranges, {"units": "m"})},
        attrs={"source": "METEK MIRA-35", "instrument_pid": "pid"},
    ).to_netcdf(path)
    return snr


def test_radar_reader_flags_minutes_at_the_ceiling(tmp_path):
    path = tmp_path / "mira.nc"
    snr = write_radar(path, with_snr=True)
    radar = read_radar_cloudnet(path)
    qf = radar["QF_saturation"]
    assert qf.dims == ("time", "range") and qf.sizes["time"] == 1440
    assert np.isfinite(radar["saturation_snr_cap"].values[0])
    assert np.isnan(radar["saturation_snr_cap"].values[1])
    assert (qf.isel(range=0) == 1).any() and not (qf.isel(range=1) != 0).any()
    # flagged minutes are those where the clamped profiles are
    times = pd.date_range("2025-05-05 00:00:02", periods=N, freq="4s")
    cap = radar["saturation_snr_cap"].values[0]
    clamped = pd.Series(snr[:, 0] >= cap - saturation.NEAR_DB, index=times)
    # same bins as the reflectivity: (t - 30 s, t + 30 s]
    minute = clamped.groupby((times - pd.Timedelta(30, "s")).ceil("1min")).mean()
    flagged = qf.isel(range=0).to_series()
    common = minute.index.intersection(flagged.index)
    expected = (minute[common] >= saturation.MINUTE_FRACTION).astype(int)
    assert (flagged[common].values == expected.values).all()


def test_radar_without_snr_is_not_assessed(tmp_path):
    path = tmp_path / "rpg.nc"
    write_radar(path, with_snr=False)
    radar = read_radar_cloudnet(path)
    assert (radar["QF_saturation"] == 0).all()
    assert np.isnan(radar["Zdcr_saturation_ceiling"].values).all()
    assert "not assessed" in radar["QF_saturation"].attrs["comment"]


def preprocessed(qf, ceiling, zdd):
    time = pd.date_range("2025-05-05", periods=len(zdd), freq="1min")
    return xr.Dataset(
        {
            "QF_saturation": (("time", "range"), np.array(qf, dtype="i1")),
            "Zdcr_saturation_ceiling": (("range",), np.array(ceiling, dtype=float)),
            "Zdlog_vfov_modv_tm": (
                ("time", "radar_frequencies"),
                np.array(zdd, dtype=float)[:, None],
            ),
            "radar_frequency": ((), 35.5e9),
        },
        coords={
            "time": time,
            "range": [150.0, 300.0],
            "radar_frequencies": [35.5e9],
        },
    )


def test_minutes_with_zdd_near_the_ceiling_are_flagged():
    ds = preprocessed(
        qf=[[0, 0], [0, 1], [0, 0]], ceiling=[np.nan, 22.0], zdd=[17.0, 25.0, 19.0]
    )
    out = saturation.flag_disdrometer_near_ceiling(ds)["QF_saturation"]
    # gate without ceiling untouched ; 150 m stays 0 even for Zdd = 25 dBZ
    assert out.isel(range=0).values.tolist() == [0, 0, 0]
    # 300 m: 17 < 22 - 4 kept ; at ceiling stays 1 ; 19 >= 18 becomes 2
    assert out.isel(range=1).values.tolist() == [0, 1, 2]


def test_qc_at_the_delta_z_gate():
    ds = preprocessed(
        qf=[[0, 0], [0, 1], [0, 2]], ceiling=[np.nan, 22.0], zdd=[0, 0, 0]
    )
    conf = {"instrument_parameters": {"DCR_DZ_RANGE": 290}}
    assert saturation.saturation_qc(ds, conf).values.tolist() == [1, 0, 0]
    conf = {"instrument_parameters": {"DCR_DZ_RANGE": 150}}
    assert saturation.saturation_qc(ds, conf).values.tolist() == [1, 1, 1]
    qc = saturation.saturation_qc(ds.drop_vars("QF_saturation"), conf)
    assert qc.values.tolist() == [1, 1, 1] and "not assessed" in qc.attrs["comment"]
