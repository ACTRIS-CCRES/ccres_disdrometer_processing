"""Tests the preprocessing speed-ups leave the results unchanged."""

import numpy as np
import pytest
import toml

from ccres_disdrometer_processing import dielectric, scattering, utils
from ccres_disdrometer_processing.open_disdro_netcdf import (
    read_parsivel_cloudnet_choice,
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

    # the largest drop is left out : at 94 GHz its T-matrix needs the fallback
    # quadrature and takes 43 s (the fallback is tested in test_scattering)
    monkeypatch.setenv(scattering.NO_TABLE_ENV, "1")
    scattering._tables.cache_clear()
    for (f, o), table in from_table.items():
        fresh = scattering.scattering_prop(D[:-1], o, f, e=water[f], axrMethod=AXR)
        for v in scattering.TABLE_VARS:
            # equal to the last bit on the machine that built the tables ; the
            # T-matrix rounds differently on other processors (up to 1.2e-6
            # relative on the GitHub runners, for the largest drops at 94 GHz ;
            # 1e-4 relative is 4e-4 dB)
            np.testing.assert_allclose(
                getattr(table, v)[:-1], getattr(fresh, v), rtol=1e-4, atol=0
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
