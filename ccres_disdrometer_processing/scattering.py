import logging

import numpy as np
from pytmatrix import radar, tmatrix_aux
from pytmatrix.tmatrix import Scatterer
from scipy import constants

lgr = logging.getLogger(__name__)


class DATA:
    def __init__(self):
        self.time = []


def compute_fallspeed(d, strMethod="GunAndKinzer"):
    # Gun and Kinzer 1949
    # d is in mm
    # v is in m/s
    if strMethod == "GunAndKinzer":
        v = 9.40 * (1 - np.exp(-1.57 * (10**3) * np.power(d * (10**-3), 1.15)))
    elif strMethod == "Khvorostyanov_Curry_2002":
        raise NotImplementedError
    elif strMethod == "Atlas_Ulbrich_1977":
        # Atlas and Ulbrich (1977) /  D is in cm and v0(r) has units of m s-1
        # v = 28.11*(d *0.1/2.)**0.67
        raise NotImplementedError
    return v


def axis_ratio(D, axrMethod="BeardChuang_PolynomialFit"):
    # describe the shape of the droplet vs. its diameter
    # Andsager et al., 1999 fit, from Beard and Chuang 1987 model
    if axrMethod == "BeardChuang_PolynomialFit":
        AR = 1.0 / (
            1.0048 + 5.7e-4 * D - 2.628e-2 * D**2 + 3.682e-3 * D**3 - 1.677e-4 * D**4
        )
        AR[AR < 0.0] = 2.2
        AR[AR > 2.2] = 2.2
        return AR


def compute_bscat_tmatrix(Diam, lambda_m, e, axis_ratio, beam_orientation):
    scatterer_tm = Scatterer(
        radius=(0.5 * Diam * 1e3), wavelength=lambda_m * 1e3, m=e, axis_ratio=axis_ratio
    )

    # Backscattering coef
    if beam_orientation == 0:
        scatterer_tm.set_geometry(tmatrix_aux.geom_horiz_back)
    else:
        scatterer_tm.set_geometry(tmatrix_aux.geom_vert_back)
    bscat_tmat = radar.refl(scatterer_tm)

    # Specific attenuation (dB/km)
    if beam_orientation == 0:
        scatterer_tm.set_geometry(tmatrix_aux.geom_horiz_forw)
    else:
        scatterer_tm.set_geometry(tmatrix_aux.geom_vert_forw)
    att_tmat = radar.Ai(scatterer_tm)
    return bscat_tmat, att_tmat


def compute_bscat_mie(Diam, lambda_m, e, beam_orientation):
    scatterer_mie = Scatterer(
        radius=(0.5 * Diam * 1e3),
        wavelength=lambda_m * 1e3,
        m=e,
        axis_ratio=1,
    )
    if beam_orientation == 0:
        scatterer_mie.set_geometry(tmatrix_aux.geom_horiz_back)
    else:
        scatterer_mie.set_geometry(tmatrix_aux.geom_vert_back)
    bscat_m = radar.refl(scatterer_mie)
    return bscat_m


def scattering_prop(
    D,
    beam_orientation=1,
    freq=95.0 * 1e9,
    e=2.99645 + 1.54866 * 1j,
    axrMethod="BeardChuang_PolynomialFit",
):
    scatt = DATA()

    scatt.bscat_mie = np.zeros(np.shape(D))
    scatt.bscat_tmatrix = np.zeros(np.shape(D))
    scatt.att_tmatrix = np.zeros(np.shape(D))

    lambda_m = constants.c / freq

    # coef_ray = 1.0e18

    AXR = axis_ratio(D, axrMethod)
    lgr.debug(AXR)
    for i in range(len(D)):
        Diam = float(D[i] * 1e-3)
        bscat_tmat, att_tmat = compute_bscat_tmatrix(
            Diam, lambda_m, e, AXR[i], beam_orientation
        )
        scatt.bscat_tmatrix[i] = bscat_tmat
        scatt.att_tmatrix[i] = att_tmat

        bscat_m = compute_bscat_mie(Diam, lambda_m, e, beam_orientation)
        bscat_m = compute_bscat_tmatrix(Diam, lambda_m, e, 1, beam_orientation)[0]
        scatt.bscat_mie[i] = bscat_m
        # lgr.debug(bscat_tmat, bscat_m)

    return scatt
