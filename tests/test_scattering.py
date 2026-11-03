"""Tests the T-matrix scattering computations (rustmatrix backend)."""

import logging

import numpy as np
import pytest
from scipy import constants

from ccres_disdrometer_processing import scattering

E_WATER = complex(2.99645, 1.54866)  # refractive index used in all station configs

# Reference values computed with pytmatrix-lte 0.4.5, the backend used before
# rustmatrix : (frequency [Hz], diameter [mm], beam orientation, bscat, att)
PYTMATRIX_REFERENCE = [
    (1.0e10, 0.562, 1, 2.492949e-02, 4.479060e-05),
    (1.0e10, 0.562, 0, 2.492906e-02, 4.479125e-05),
    (1.0e10, 1.875, 1, 3.629602e01, 1.953141e-03),
    (1.0e10, 1.875, 0, 3.612084e01, 1.960792e-03),
    (1.0e10, 3.25, 1, 1.111637e03, 1.420511e-02),
    (1.0e10, 3.25, 0, 1.072015e03, 1.457941e-02),
    (1.0e10, 4.75, 1, 1.328271e04, 7.262649e-02),
    (1.0e10, 4.75, 0, 1.175586e04, 7.784459e-02),
    (9.4e10, 0.562, 1, 2.593294e-02, 1.104520e-03),
    (9.4e10, 0.562, 0, 2.590089e-02, 1.105228e-03),
    (9.4e10, 1.875, 1, 2.683726e-01, 3.693438e-02),
    (9.4e10, 1.875, 0, 3.369609e-01, 3.656049e-02),
    (9.4e10, 3.25, 1, 1.494659e00, 1.056845e-01),
    (9.4e10, 3.25, 0, 1.497610e00, 9.965675e-02),
    (9.4e10, 4.75, 1, 5.275489e00, 2.297230e-01),
    (9.4e10, 4.75, 0, 1.041165e00, 1.979401e-01),
]


@pytest.mark.parametrize(("freq", "d_mm", "fov", "bscat", "att"), PYTMATRIX_REFERENCE)
def test_tmatrix_matches_pytmatrix(freq, d_mm, fov, bscat, att):
    axis_ratio = scattering.axis_ratio(np.array([d_mm]))[0]
    got_bscat, got_att = scattering.compute_bscat_tmatrix(
        d_mm * 1e-3, constants.c / freq, E_WATER, axis_ratio, fov
    )
    # 0.5 % on bscat is about 0.02 dB
    assert got_bscat == pytest.approx(bscat, rel=5e-3)
    assert got_att == pytest.approx(att, rel=5e-3)


def test_mie_is_the_spherical_tmatrix():
    lambda_m = constants.c / 35e9
    for fov in (0, 1):
        mie = scattering.compute_bscat_mie(2e-3, lambda_m, E_WATER, fov)
        sphere = scattering.compute_bscat_tmatrix(2e-3, lambda_m, E_WATER, 1.0, fov)
        assert mie == sphere[0]


def test_tmatrix_falls_back_when_the_solver_fails(caplog):
    # 13 mm drop with the maximum axis ratio at 94 GHz : rustmatrix fails with
    # the default quadrature (and pytmatrix-lte does not converge there either)
    lambda_m = constants.c / 94e9
    scattering._scatterer.cache_clear()  # the failure is only met once per process
    with caplog.at_level(logging.WARNING, logger=scattering.lgr.name):
        bscat, att = scattering.compute_bscat_tmatrix(13e-3, lambda_m, E_WATER, 2.2, 1)
    assert np.isfinite(bscat) and bscat > 0
    assert np.isfinite(att) and att > 0
    assert f"ndgs = {scattering.NDGS_FALLBACK}" in caplog.text
