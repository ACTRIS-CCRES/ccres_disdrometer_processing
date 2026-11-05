"""Refractive index and dielectric factor of liquid water at the radar frequency.

The scattering of the drops is computed at the radar's own frequency, read in the
radar file, with :

- the refractive index of water from Ellison (2007) at T_REF, the temperature of
  the drops : a single compromise for every station (the wet-bulb temperature
  during rain is 8 to 10 C at the European sites, and Zdd changes by only 0.03 to
  0.06 dB per C) ;
- the dielectric factor |K|^2 the radar uses to convert its power into Ze, so that
  Zdd follows the same convention as Zdcr : 0.93 in X band, 0.88 in Ka band (the
  Cloudnet convention), 0.74 in W band (|K| = 0.86, used by the RPG software and
  for the BASTA, see Toledo and Delanoe).

Both are read in a fixed table (assets/water_dielectric.toml, written by
scripts/build_dielectric_table.py) for the frequencies of the network, so that
nothing is recomputed at each run. A frequency absent from the table is computed
with the same formulas, with a warning : add it to the table.
"""

import cmath
import functools
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import toml

lgr = logging.getLogger(__name__)

TABLE_FILE = Path(__file__).parent / "assets" / "water_dielectric.toml"

T_REF = 10.0  # C, temperature of the drops for the refractive index
# a radar frequency within this distance of a table entry uses that entry
FREQUENCY_TOLERANCE = 0.02e9  # Hz
# |K|^2 by band, (upper frequency bound in Hz, |K|^2), see the module docstring
KW_SQR_BANDS = ((15e9, 0.93), (50e9, 0.88), (math.inf, 0.74))
# |K|^2 rustmatrix normalises its reflectivities with (Scatterer.Kw_sqr default)
RUSTMATRIX_KW_SQR = 0.93


# Ellison (2007), J. Phys. Chem. Ref. Data 36, 1 : Eqs. (17a)-(17b) with the
# "best fit" coefficients of Table 2, valid for 0-100 C and 0-25 THz.
# Three Debye relaxation terms : Delta_i(t) = a_i exp(-b_i t),
# tau_i(t) = c_i exp(d_i / (t + t_c))
_A = (79.23882, 3.815866, 1.634967)
_B = (0.004300598, 0.01117295, 0.006841548)
_C = (1.382264e-13, 3.510354e-16, 6.30035e-15)
_D = (652.7648, 1249.533, 405.5169)
_T_C = 133.1383
# first resonance : Delta_4(t), f0(t), tau_4(t)
_P_DELTA4 = (0.8379692, -0.006118594, -0.000012936798)
_P_F0 = (4235901000000.0, -14260880000.0, 273815700.0, -1246943.0)
_P_TAU4 = (9.618642e-14, 1.795786e-16, -9.310017e-18, 1.655473e-19)
# second resonance : Delta_5(t), f1(t), tau_5(t)
_P_DELTA5 = (0.6165532, 0.007238532, -0.00009523366)
_P_F1 = (15983170000000.0, -74413570000.0, 497448000.0)
_P_TAU5 = (2.882476e-14, -3.142118e-16, 3.528051e-18)


def _poly(p, t):
    return sum(c * t**i for i, c in enumerate(p))


def ellison_permittivity(freq_hz: float, temp_c: float = T_REF) -> complex:
    """Return the complex permittivity of pure water, Ellison (2007).

    Parameters
    ----------
    freq_hz : float
        Frequency (Hz), 0 to 25 THz
    temp_c : float, optional
        Temperature (C), 0 to 100, by default T_REF

    Returns
    -------
    complex
        eps' + i eps''

    """
    nu, t = float(freq_hz), float(temp_c)
    w = 2 * math.pi * nu
    eps_s = 87.9144 - 0.404399 * t + 9.58726e-4 * t**2 - 1.32802e-6 * t**3
    real, imag = eps_s, 0.0
    for a, b, c, d in zip(_A, _B, _C, _D):  # noqa: B905
        delta = a * math.exp(-b * t)
        tau = c * math.exp(d / (t + _T_C))
        real -= w**2 * tau**2 * delta / (1 + (w * tau) ** 2)
        imag += w * tau * delta / (1 + (w * tau) ** 2)
    for delta, f, tau in (
        (_poly(_P_DELTA4, t), _poly(_P_F0, t), _poly(_P_TAU4, t)),
        (_poly(_P_DELTA5, t), _poly(_P_F1, t), _poly(_P_TAU5, t)),
    ):
        lo = 1 + (2 * math.pi * tau * (f + nu)) ** 2
        hi = 1 + (2 * math.pi * tau * (f - nu)) ** 2
        real -= (2 * math.pi * tau) ** 2 * delta / 2 * ((f + nu) / lo - (f - nu) / hi)
        imag += math.pi * nu * tau * delta * (1 / lo + 1 / hi)
    return complex(real, imag)


def ellison_refractive_index(freq_hz: float, temp_c: float = T_REF) -> complex:
    """Return the complex refractive index n + ik of water, Ellison (2007)."""
    return cmath.sqrt(ellison_permittivity(freq_hz, temp_c))


def kw_sqr_convention(freq_hz: float) -> float:
    """Return the |K|^2 the radars of this band use to compute Ze."""
    for upper, kw in KW_SQR_BANDS:
        if freq_hz < upper:
            return kw
    raise ValueError(f"invalid frequency {freq_hz}")


@dataclass(frozen=True)
class WaterDielectric:
    """Water properties used for the scattering at one radar frequency."""

    frequency: float  # Hz, frequency of the scattering computation
    refractive_index: complex
    kw_sqr: float
    temperature: float  # C
    from_table: bool


@functools.cache
def _table():
    """Return the fixed table, {frequency (Hz): WaterDielectric} (cached)."""
    if not TABLE_FILE.exists():
        return {}
    content = toml.load(TABLE_FILE)
    out = {}
    for e in content["entry"]:
        f = float(e["frequency_ghz"]) * 1e9
        out[f] = WaterDielectric(
            frequency=f,
            refractive_index=complex(*e["refractive_index"]),
            kw_sqr=float(e["kw_sqr"]),
            temperature=float(content["temperature"]),
            from_table=True,
        )
    return out


def water_dielectric(radar_frequency: float) -> WaterDielectric:
    """Return the water properties for the scattering at a radar frequency.

    The table entry within FREQUENCY_TOLERANCE of the radar frequency, the
    scattering being then computed at that entry's frequency (this keeps the
    precomputed scattering tables usable : an error below 0.01 dB). Otherwise the
    radar frequency rounded to 10 MHz, with the index computed from Ellison.

    Parameters
    ----------
    radar_frequency : float
        Frequency of the radar (Hz)

    Returns
    -------
    WaterDielectric

    """
    f = float(radar_frequency)
    if not math.isfinite(f) or f <= 0:
        raise ValueError(
            f"invalid radar frequency {radar_frequency} : the scattering is "
            "computed at the radar's frequency, which must be in the radar file"
        )
    table = _table()
    if table:
        nearest = min(table, key=lambda x: abs(x - f))
        if abs(nearest - f) <= FREQUENCY_TOLERANCE:
            return table[nearest]
    f = round(f / 1e7) * 1e7
    lgr.warning(
        f"Radar frequency {f / 1e9:.2f} GHz not in {TABLE_FILE.name} : water "
        "refractive index computed (add it to the table, and to the scattering "
        "tables, see scripts/build_dielectric_table.py)"
    )
    return WaterDielectric(
        frequency=f,
        refractive_index=ellison_refractive_index(f, T_REF),
        kw_sqr=kw_sqr_convention(f),
        temperature=T_REF,
        from_table=False,
    )
