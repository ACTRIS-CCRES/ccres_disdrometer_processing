"""Tests the preprocessing speed-ups leave the results unchanged."""

import numpy as np
import pytest
import toml

from ccres_disdrometer_processing import dielectric, scattering, utils
from ccres_disdrometer_processing.open_disdro_netcdf import (
    read_parsivel_cloudnet_choice,
    reflectivity_model_multilambda_measmodV_hvfov,
)

E = 2.99645 + 1.54866j  # former refractive index of the configurations
AXR = "BeardChuang_PolynomialFit"


@pytest.fixture
def juelich(data_input_dir, data_conf_dir):
    source = utils.get_file_from_cloudnet(
        "juelich",
        "2024-02-08",
        "parsivel",
        "https://hdl.handle.net/21.12132/3.2a1ca46ed70c4929",
        data_input_dir,
    )
    config = toml.load(data_conf_dir / "config_juelich_mira-parsivel.toml")
    return read_parsivel_cloudnet_choice(source, [35.0e9, 94.0e9], config)


@pytest.fixture
def computed(monkeypatch):
    """Ignore the precomputed tables : the scattering is computed."""
    monkeypatch.setenv(scattering.NO_TABLE_ENV, "1")
    scattering._tables.cache_clear()
    yield
    scattering._tables.cache_clear()


def test_tables_hold_the_computed_values(juelich, monkeypatch):
    D = juelich.size_classes[0:-5]
    scattering._tables.cache_clear()
    water = {f: dielectric.water_dielectric(f).refractive_index for f in (35e9, 94e9)}
    from_table = {
        (f, o): scattering._from_table(D, o, f, water[f], AXR)
        for f in (35e9, 94e9)
        for o in (0, 1)
    }
    assert all(v is not None for v in from_table.values())

    monkeypatch.setenv(scattering.NO_TABLE_ENV, "1")
    scattering._tables.cache_clear()
    for (f, o), table in from_table.items():
        fresh = scattering.scattering_prop(D, o, f, e=water[f], axrMethod=AXR)
        for v in scattering.TABLE_VARS:
            # equal to the last bit on the machine that built the tables ; the
            # T-matrix rounds differently on other processors (up to 1.2e-6
            # relative on the GitHub runners, for the largest drops at 94 GHz ;
            # 1e-4 relative is 4e-4 dB)
            np.testing.assert_allclose(
                getattr(table, v), getattr(fresh, v), rtol=1e-4, atol=0
            )
    scattering._tables.cache_clear()


def test_every_network_frequency_is_in_the_tables(juelich):
    D = juelich.size_classes[0:-5]
    scattering._tables.cache_clear()
    for w in dielectric._table().values():
        for o in (0, 1):
            assert scattering._from_table(D, o, w.frequency, w.refractive_index, AXR)


def test_a_case_absent_from_the_tables_is_computed(juelich):
    D = juelich.size_classes[0:-5]
    e = dielectric.water_dielectric(35e9).refractive_index
    scattering._tables.cache_clear()
    assert scattering._from_table(D, 1, 35e9, e, AXR) is not None
    assert scattering._from_table(D, 1, 77e9, e, AXR) is None
    assert scattering._from_table(D, 1, 35e9, E, AXR) is None  # former index
    assert scattering._from_table(D[:-1], 1, 35e9, e, AXR) is None
    assert scattering._from_table(D.astype(float), 1, 35e9, e, AXR) is None
    s = scattering.scattering_prop(D[:3], 1, 77e9, e=E, axrMethod=AXR)
    assert np.all(s.bscat_tmatrix > 0)


def reference_model(m, scatt_list, n):
    """Compute with the former per-timestep loop of the reflectivity model."""
    psd, D, V = m.psd.values, m.size_classes.values, m.speed_classes.values
    F, t_all = m.F.data, m.disdro_interval.values.astype(float)
    VDmodel = scattering.compute_fallspeed(D)
    nt, fr = len(m.time), len(m.radar_frequencies)
    out = {k: np.zeros((nt, 2)) for k in ("M2", "M3", "Ze_ray")}
    out["RR"], out["VD"] = np.zeros(nt), np.zeros((nt, len(D)))
    for k in ("Ze_mie", "Ze_tm", "attenuation", "V_mie", "V_tm"):
        out[k] = np.zeros((nt, fr, 2, 2))
    for ii in range(nt):
        t = t_all[ii]
        Ni = np.nansum(psd[ii, :, :], 1)
        out["RR"][ii] = (
            (np.pi / 6.0) * (1.0 / (F * t)) * np.nansum(Ni * (D**3)) * (3.6 * 1e-3)
        )
        for i in range(len(D)):
            out["VD"][ii, i] = np.nansum(psd[ii, i, :] * V) / np.nansum(psd[ii, i, :])
        VDD = np.vstack((out["VD"][ii, :], VDmodel)).T
        for k, x in (
            ("M2", (D * 1.0e-3) ** 2),
            ("M3", (D * 1.0e-3) ** 4),
            ("Ze_ray", D**6),
        ):
            out[k][ii, :] = np.nansum(np.tile(Ni * x, (2, 1)).T / VDD, axis=0) / F / t
        for k, sc in enumerate(scatt_list):
            f, o = k % fr, k // fr
            for name, b in (
                ("Ze_mie", sc.bscat_mie),
                ("Ze_tm", sc.bscat_tmatrix),
                ("attenuation", sc.att_tmatrix),
            ):
                out[name][ii, f, o, :] = (
                    np.nansum(np.tile(Ni[0:n] * b, (2, 1)).T / VDD[0:n, :], axis=0)
                    / F
                    / t
                )
            out["Ze_tm"][np.where(out["Ze_tm"] == 0)] = np.nan
            for name, b in (("V_mie", sc.bscat_mie), ("V_tm", sc.bscat_tmatrix)):
                out[name][ii, f, o, :] = np.nansum(Ni[0:n] * b) / np.nansum(
                    np.tile(Ni[0:n] * b, (2, 1)).T / VDD[0:n, :], axis=0
                )
    return out


def test_vectorised_model_matches_the_former_loop(juelich):
    m = juelich.isel(time=slice(0, 400))  # rain in the first hours of the day
    n = len(m.size_classes[0:-5])
    scatt = [
        scattering.scattering_prop(m.size_classes[0:-5], o, f, e=E, axrMethod=AXR)
        for o in (1, 0)
        for f in (35.0e9, 94.0e9)
    ]
    ref = reference_model(m, scatt, n)
    out = reflectivity_model_multilambda_measmodV_hvfov(m.copy(), scatt, n)
    assert np.isfinite(ref["Ze_tm"]).sum() > 100
    pairs = {
        "disdro_pr_from_raw": ref["RR"],
        "measV": ref["VD"],
        "Zdlin_vfov_modv_tm": ref["Ze_tm"][:, :, 0, 1],
        "Zdlin_hfov_measv_tm": ref["Ze_tm"][:, :, 1, 0],
        "Zdlin_vfov_measv_mie": ref["Ze_mie"][:, :, 0, 0],
        "Zdlin_modv_ray": ref["Ze_ray"][:, 1],
        "attd_vfov_modv": ref["attenuation"][:, :, 0, 1],
        "DVd_vfov_modv_tm": ref["V_tm"][:, :, 0, 1],
        "DVd_hfov_measv_mie": ref["V_mie"][:, :, 1, 0],
        "m2_measv": ref["M2"][:, 0],
    }
    for name, expected in pairs.items():
        np.testing.assert_array_equal(out[name].values, expected, err_msg=name)
