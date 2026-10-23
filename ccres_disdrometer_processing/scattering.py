import contextlib
import functools
import logging
import os
import sys

import numpy as np
from rustmatrix import Scatterer, radar, tmatrix_aux
from scipy import constants

lgr = logging.getLogger(__name__)

# T-matrix quadrature setting (number of division points per expansion order) :
# NDGS is the default of rustmatrix and pytmatrix, NDGS_FALLBACK is used when the
# solver fails with it (giant oblate drops at 94 GHz exceed its quadrature limit)
NDGS = 2
NDGS_FALLBACK = 1


class DATA:
    def __init__(self):
        self.time = []


@contextlib.contextmanager
def _stderr_hidden():
    """Hide what is written to the process stderr (file descriptor 2).

    A rustmatrix solver failure is a Rust panic, which prints its message on
    stderr even when Python catches it.
    """
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


@functools.cache
def _scatterer(Diam, lambda_m, e, axis_ratio):
    """Return a drop Scatterer whose T-matrix is computed, cached in the process.

    The T-matrix only depends on the drop and the wavelength, not on the beam
    geometry : both beam orientations, backscattering and attenuation reuse it.
    When the solver fails with NDGS (giant oblate drops at 94 GHz exceed its
    quadrature limit), the T-matrix is computed again with NDGS_FALLBACK.

    Parameters
    ----------
    Diam : float
        Diameter of the droplet (m)
    lambda_m : float
        wavelength of the radar wave (m)
    e : complex
        Complex refractive index of water
    axis_ratio : float
        droplet axis ratio (horizontal / vertical)

    Returns
    -------
    rustmatrix.Scatterer
        The scatterer, its geometry being set by the caller before each use

    """

    def build(ndgs):
        scatterer = Scatterer(
            radius=(0.5 * Diam * 1e3),
            wavelength=lambda_m * 1e3,
            m=e,
            axis_ratio=axis_ratio,
            ndgs=ndgs,
        )
        scatterer.get_S()  # computes the T-matrix
        return scatterer

    try:
        with _stderr_hidden():
            return build(NDGS)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as err:  # a solver failure is a Rust panic, not an Exception
        lgr.warning(
            f"T-matrix failed for D = {Diam * 1e3:.3f} mm, wavelength = "
            f"{lambda_m * 1e3:.2f} mm, axis ratio = {axis_ratio:.2f} ({err}) : "
            f"computed again with ndgs = {NDGS_FALLBACK}"
        )
        return build(NDGS_FALLBACK)


def compute_fallspeed(d, strMethod="GunAndKinzer"):
    """Compute fall speed at given diameters with formulas from literature.

    Parameters
    ----------
    d : np.ndarray
        Array containing the droplet diameters (in mm) for which to compute fall speed
    strMethod : str, optional
        Formula used for fall speed computation, by default "GunAndKinzer".
        This default formula is the one described in Gun and Kinzer (1949)

    Returns
    -------
    np.ndarray
    Array containing the modeled fall speed at the specified diameters (in m/s)

    Raises
    ------
    NotImplementedError
        Formula from Khvorostyanov and Curry (2002) is yet not implemented.
    NotImplementedError
        Formula from Atlas and Ulbrich (1977) is yet not implemented.

    """
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
    """Compute the shape of the droplet knowing its diameter.

    Parameters
    ----------
    D : np.ndarray
        Array containing the diameters for which to compute axis ratio.
    axrMethod : str, optional
        Formula used for axis ratio computation, by default "BeardChuang_PolynomialFit".
        This default formula is the one described in Andsager et al. (1999),
        polynomial fit from formula by Beard and Chuang (1987)

    Returns
    -------
    np.ndarray
        Array containing the modeled axis ratios at the specified diameters.

    """
    if axrMethod == "BeardChuang_PolynomialFit":
        AR = 1.0 / (
            1.0048 + 5.7e-4 * D - 2.628e-2 * D**2 + 3.682e-3 * D**3 - 1.677e-4 * D**4
        )
        AR[AR < 0.0] = 2.2
        AR[AR > 2.2] = 2.2
        return AR


def compute_bscat_tmatrix(Diam, lambda_m, e, axis_ratio, beam_orientation):
    """Compute t-m backscattering coeff for a scatterer with known properties.

    Input a scatterer described by its diameter, axis ratio,
    with given radar frequency and index of refraction for water,
    and specified incidence of the beam.
    Output the t-matrix backscattering coefficient.

    Parameters
    ----------
    Diam : float
        Diameter of the considered droplet
    lambda_m : float
        wavelength of the radar wave
    e : complex
        Complex refractive index of water
    axis_ratio : float
        droplet axis ratio
    beam_orientation : int
        0 or 1 : orientation of the beam relative to the droplet shape

    Returns
    -------
    float, float
        backscattering coefficient and attenuation.

    """
    scatterer_tm = _scatterer(float(Diam), float(lambda_m), e, float(axis_ratio))

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
    """Compute Mie backscattering coeff for a scatterer with known properties.

    Input a scatterer described by its diameter,
    with given radar frequency and index of refraction for water,
    and specified incidence of the beam.
    For Mie backscattering, droplets are assumed to be spherical,
    i.e. axis ratio = 1
    Output the Mie backscattering coefficient.

    Parameters
    ----------
    Diam : float
        Diameter of the considered droplet
    lambda_m : float
        wavelength of the radar wave
    e : complex
        Complex refractive index of water
    beam_orientation : int
        0 or 1 : orientation of the beam relative to the droplet shape

    Returns
    -------
    float, float
        backscattering coefficient and attenuation.

    """
    scatterer_mie = _scatterer(float(Diam), float(lambda_m), e, 1.0)
    if beam_orientation == 0:
        scatterer_mie.set_geometry(tmatrix_aux.geom_horiz_back)
    else:
        scatterer_mie.set_geometry(tmatrix_aux.geom_vert_back)
    bscat_m = radar.refl(scatterer_mie)
    return bscat_m


def scattering_prop(
    D,
    beam_orientation,
    freq,
    e=2.99645 + 1.54866 * 1j,
    axrMethod="BeardChuang_PolynomialFit",
):
    """Compute attenuation, backscattering coeffs for different diameters.

    Input a vector containing all the diameters at which to compute backscattering ;
    Input the radiation characteristics ;
    Output vectors for attenuation and both Mie and t-matrix backscattering coeffs.

    Parameters
    ----------
    D : np.ndarray
        vector containing the diameters at which to compute scattering.
    beam_orientation : int, optional
        0 or 1 : orientation of the beam relative to the droplet shape, by default 1
    freq : float
        incident radar beam frequency
    e : complex, optional
        Complex refractive index of water, by default 2.99645+1.54866*1j
    axrMethod : str, optional
        formula to be used for axis ratio computation,
        by default "BeardChuang_PolynomialFit"

    Returns
    -------
    DATA
        an object containing three arrays of length (D,) :
        - one for t-matrix attenuation,
        - one for t-matrix backscattering coefficients,
        - one for Mie backscattering coefficients.

    """
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

        # sphere (axis ratio 1) : same scatterer as a T-matrix call with axis ratio 1
        scatt.bscat_mie[i] = compute_bscat_mie(Diam, lambda_m, e, beam_orientation)

    return scatt
