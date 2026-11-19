"""Keys of the station configuration file (TOML).

Every parameter of the preprocessing, the processing, the quicklooks and
gate_check is read in the station file : the code holds no default value. Each
command calls check_config before reading any data, so that a file missing a
key, or still holding an obsolete one, is reported at once, every key listed,
instead of failing in the middle of a run.
"""

from collections.abc import Mapping
from pathlib import Path

import toml

# keys the file must hold, by section
REQUIRED = {
    "location": ("SITE", "STATION"),
    "methods": ("FALL_SPEED_METHOD", "AXIS_RATIO_METHOD", "MAX_ALTITUDE_RADAR_DATA"),
    "instrument_parameters": (
        "DD_SAMPLING_AREA",
        "DCR_DZ_RANGE",
        "PROCESSING_RANGES",
        "DD_ORIENTATION",
        "DD_TIMESTAMP_CONVENTION",
        "KW_SQR",
    ),
    "thresholds": (
        "MAX_RR",
        "MIN_RAINFALL_AMOUNT",
        "MAX_WS",
        "MIN_TEMP",
        "MIN_HUR",
        "MAX_HUR",
        "DD_ANGLE",
        "MAX_INTERVAL",
        "MIN_DURATION",
        "DD_RG_MAX_PR_ACC_RATIO",
        "DD_FALLSPEED_RATIO",
        "MIN_POINTS",
    ),
    "disdrometer_interval": (
        "NOMINAL_INTERVAL",
        "MAX_TELEGRAM_INTERVAL",
        "MIN_FULL_INTERVAL",
    ),
    "disdrometer_axis_check": ("MIN_DROPS", "RATIO"),
    "radar_alignment": (
        "FALL_TIME_FACTOR",
        "MIN_FALL_SPEED",
        "MIN_WINDOW_COVERAGE",
        "RADAR_WINDOW",
    ),
    "saturation": (
        "SNR_CAP_MIN",
        "ZDD_MARGIN",
        "MINUTE_FRACTION",
        "CAP_RANK",
        "NEAR_DB",
        "BAND_DB",
        "MIN_PROFILES",
        "N_STRONG",
        "N_MIN",
        "RATIO_MIN",
    ),
    "gate_check": (
        "R_MAX",
        "R_SEARCH",
        "R_REF",
        "TOL",
        "MIN_EVENTS",
        "MIN_EVENTS_PERIOD",
        "N_BOOT",
        "BOOT_QUANTILE",
        "DV_RATIO",
        "levels",
    ),
}
# keys allowed but not read : instrument information kept in the file
INFORMATIVE = {"instrument_parameters": ("AU", "DD_SAMPLING_AREA_DEFAULT")}
# sections whose content is free
FREE_SECTIONS = ("nc_meta", "plot_parameters")
TOP_LEVEL = ("title",)
# thresholds of each stratiform selection level of gate_check
GATE_LEVEL_KEYS = (
    "n_qc",
    "rain_frac",
    "qc_frac",
    "lpr_std",
    "pr_p95",
    "z5_jump",
    "dv_ratio",
    "dv_lo",
    "z_grad",
)
TIMESTAMP_CONVENTIONS = ("end", "start")


class ConfigError(ValueError):
    """The station configuration file lacks keys or holds invalid values."""


def config_problems(conf: Mapping) -> list[str]:
    """Return the problems of a station configuration, empty if none."""
    problems = []
    for section, keys in REQUIRED.items():
        values = conf.get(section)
        if not isinstance(values, Mapping):
            problems.append(f"missing section [{section}]")
            continue
        problems += [f"missing key {k} in [{section}]" for k in keys if k not in values]
        known = set(keys) | set(INFORMATIVE.get(section, ()))
        problems += [
            f"unknown key {k} in [{section}]" for k in values if k not in known
        ]
    for name, value in conf.items():
        if name in REQUIRED or name in FREE_SECTIONS:
            continue
        if not (name in TOP_LEVEL and not isinstance(value, Mapping)):
            problems.append(f"unknown section or key {name}")
    if problems:
        return problems

    ip = conf["instrument_parameters"]
    if ip["DD_TIMESTAMP_CONVENTION"] not in TIMESTAMP_CONVENTIONS:
        problems.append(
            f"DD_TIMESTAMP_CONVENTION = {ip['DD_TIMESTAMP_CONVENTION']!r} not in "
            f"{TIMESTAMP_CONVENTIONS}"
        )
    if not 0 < ip["KW_SQR"] <= 1:
        problems.append(f"KW_SQR = {ip['KW_SQR']} outside (0, 1]")
    band = conf["saturation"]["BAND_DB"]
    if len(band) != 2 or not 0 <= band[0] < band[1]:
        problems.append(f"BAND_DB = {band} : two increasing values expected (dB)")
    levels = conf["gate_check"]["levels"]
    if not isinstance(levels, Mapping) or not levels:
        problems.append("[gate_check.levels] : at least one level expected")
    else:
        for name, th in levels.items():
            missing = [k for k in GATE_LEVEL_KEYS if k not in th]
            unknown = [k for k in th if k not in GATE_LEVEL_KEYS]
            if missing:
                problems.append(
                    f"[gate_check.levels.{name}] lacks {', '.join(missing)}"
                )
            if unknown:
                problems.append(
                    f"[gate_check.levels.{name}] : unknown {', '.join(unknown)}"
                )
    return problems


def check_config(conf: Mapping, source="station configuration") -> Mapping:
    """Raise ConfigError listing every problem of the configuration, else return it."""
    problems = config_problems(conf)
    if problems:
        raise ConfigError(
            f"{source} : " + "; ".join(problems) + " (see the station files of "
            "ccres_disdro_config for the expected sections)"
        )
    return conf


def load_config(path) -> dict:
    """Read and check a station configuration file."""
    return dict(check_config(toml.load(path), source=Path(path).name))
