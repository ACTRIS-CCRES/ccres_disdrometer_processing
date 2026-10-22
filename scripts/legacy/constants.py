"""Module containing physical constants."""

from scipy import constants

FREQ = 95.0 * 1e9  # Hz
FREQ_35 = 35.0 * 1e9
FREQ_94 = 94.0 * 1e9
E = 2.99645 + 1.54866 * 1j
LAMBDA_M = constants.c / FREQ
F_PARSIVEL = 0.0054  # m2, sampling surface
AUparameter_default = 1000
AUparameter_lindenberg = 1009  # 1009 @ Lindenberg !
AUparameter_palaiseau = 965  # @ SIRTA
F_THIES = 0.0046 * 1000 / AUparameter_default
