"""Tests the optimal gate estimation of gate_check."""

import numpy as np
import pandas as pd
import xarray as xr

from ccres_disdrometer_processing import gate_check


def profiles(excess_top, n_events=20, noise=0.05, seed=0):
    """Deviation profiles with an excess decreasing up to excess_top (m)."""
    gates = np.arange(100.0, 601.0, 25.0)
    rng = np.random.default_rng(seed)
    base = np.where(gates < excess_top, 2.0 * (excess_top - gates) / 100, 0.0)
    return gates, base + rng.normal(0, noise, (n_events, gates.size))


def test_optimal_gate_is_the_first_one_within_the_tolerance():
    gates, devs = profiles(excess_top=300.0)
    g, comp = gate_check.optimal_gate(gates, devs)
    # 2 dB/100 m : the excess falls below 0.25 dB above 287.5 m
    assert g == 300.0
    assert abs(comp[gates == 300.0][0]) <= gate_check.TOL
    # a deficit (incomplete overlap) counts as well
    g, _ = gate_check.optimal_gate(gates, -devs)
    assert g == 300.0


def test_bootstrap_and_recommendation():
    gates, devs = profiles(excess_top=250.0)
    boot = gate_check.bootstrap_gates(gates, devs, n_boot=50)
    assert np.all(boot == 250.0)
    row, _ = gate_check.summarise(gates, devs, conf_gate=150.0, label="test")
    assert row["gate_recommended"] == 250.0
    assert row["dev_at_conf_gate"] > 1.5


def test_deviation_from_the_reference_line():
    r = np.arange(100.0, 601.0, 25.0)
    medc = -1.0 - 0.002 * r + np.where(r < 200, 1.0, 0.0)  # sloped line + excess
    dev, (a, b) = gate_check.deviation(
        r, medc, np.full(r.size, 100), np.full(r.size, 5)
    )
    assert abs(a + 0.002) < 1e-4  # 3-gate running median, 2 gates at the top
    np.testing.assert_allclose(dev[r >= 200], 0.0, atol=0.02)
    assert np.all(dev[r < 200] > 0.9)
    # a melting layer (slow fall speed) is left out of the fit
    v = np.where(r > 500, 1.0, 5.0)
    dev, _ = gate_check.deviation(r, medc, np.full(r.size, 100), v)
    assert np.isnan(dev[r > 500]).all()


def test_selection_level_takes_the_most_demanding_with_enough_events():
    strict = {
        k: (v + 1 if k in gate_check.MINIMA else v - 0.01)
        for k, v in gate_check.LEVELS["strict"].items()
    }
    large_only = {
        k: (v if k in gate_check.MINIMA else v)
        for k, v in gate_check.LEVELS["large"].items()
    }
    f = pd.DataFrame([strict] * 16 + [large_only] * 10)
    passes, level = gate_check.selection_level(f, min_events=15)
    assert level == "strict"
    assert passes["large"].sum() == 26
    _, level = gate_check.selection_level(f.iloc[10:], min_events=15)
    assert level == "large"


def test_bistatic_radars_take_the_gate_above_the_first_one_within_tolerance():
    gates, devs = profiles(excess_top=250.0)
    mono, _ = gate_check.summarise(gates, -devs, conf_gate=150.0, label="mono")
    bi, _ = gate_check.summarise(
        gates, -devs, conf_gate=150.0, label="bi", bistatic=True
    )
    assert mono["gate_recommended"] == 250.0
    assert np.isnan(mono["gate_composite_plus_one"])
    assert bi["gate_composite"] == 250.0
    assert bi["gate_composite_plus_one"] == 275.0
    assert bi["gate_recommended"] == 275.0
    assert gate_check.antenna_layout("juelich_mira-35_0366fa69_parsivel_2a1ca46e") == (
        "monostatique"
    )
    for combo in (
        "juelich_rpg-fmcw-94_f7e00266_parsivel_2a1ca46e",
        "palaiseau_basta_643b7b5b_parsivel_7e13f3f2",
    ):
        assert gate_check.antenna_layout(combo) == "bistatique"


def test_a_minute_saturated_at_any_gate_up_to_the_reference_layer_is_dropped():
    r = np.arange(150.0, 1001.0, 50.0)
    qf = np.zeros((4, r.size), dtype="i1")
    qf[1, r == 400.0] = 1  # radar at its ceiling in the reference layer
    qf[2, r == 200.0] = 2  # Zdd near the ceiling below it
    qf[3, r == 900.0] = 1  # above R_SEARCH : kept
    ds = xr.Dataset({"QF_saturation": (("time", "range"), qf)}, coords={"range": r})
    sat = gate_check.saturated_minutes(ds)
    assert sat.tolist() == [False, True, True, False]
    assert gate_check.saturated_minutes(ds.drop_vars("QF_saturation")) is None
