"""Module containing physical constants."""
from scipy import constants

FREQ = 95.0 * 1e9  # Hz
BEAM_ORIENTATION = 1
E = 2.99645 + 1.54866 * 1j
LAMBDA_M = constants.c / FREQ
F_PARSIVEL = 0.0054  # m2, sampling surface
