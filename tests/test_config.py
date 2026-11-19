"""Tests the check of the station configuration files."""

import copy

import pytest
import toml

from ccres_disdrometer_processing.config import (
    ConfigError,
    check_config,
    config_problems,
    load_config,
)
from tests.station_conf import CONF_FILE


@pytest.fixture
def conf():
    return copy.deepcopy(toml.load(CONF_FILE))


def test_every_test_configuration_is_complete(data_conf_dir):
    files = sorted(data_conf_dir.glob("*.toml"))
    assert files
    for f in files:
        load_config(f)


def test_every_missing_key_is_listed(conf):
    del conf["saturation"]["ZDD_MARGIN"]
    del conf["radar_alignment"]["FALL_TIME_FACTOR"]
    del conf["instrument_parameters"]["KW_SQR"]
    problems = config_problems(conf)
    assert "missing key ZDD_MARGIN in [saturation]" in problems
    assert "missing key FALL_TIME_FACTOR in [radar_alignment]" in problems
    assert "missing key KW_SQR in [instrument_parameters]" in problems
    with pytest.raises(ConfigError, match="KW_SQR.*FALL_TIME_FACTOR.*ZDD_MARGIN"):
        check_config(conf)


def test_a_missing_section_is_reported(conf):
    del conf["gate_check"]
    assert config_problems(conf) == ["missing section [gate_check]"]


def test_obsolete_or_misplaced_keys_are_reported(conf):
    conf["methods"]["REFRACTION_INDEX"] = [[2.99645, 1.54866]]
    conf["methods"]["FALL_TIME_FACTOR"] = 0.55
    conf["thresholds"]["PR_SAMPLING"] = 15
    conf["extra"] = {}
    problems = config_problems(conf)
    assert "unknown key REFRACTION_INDEX in [methods]" in problems
    assert "unknown key FALL_TIME_FACTOR in [methods]" in problems
    assert "unknown key PR_SAMPLING in [thresholds]" in problems
    assert "unknown section or key extra" in problems


def test_informative_and_free_keys_are_accepted(conf):
    conf["instrument_parameters"]["AU"] = 1009
    conf["nc_meta"]["anything"] = ""
    assert config_problems(conf) == []


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("instrument_parameters", "DD_TIMESTAMP_CONVENTION", "middle", "not in"),
        ("instrument_parameters", "KW_SQR", 0.0, "outside"),
        ("instrument_parameters", "KW_SQR", 1.2, "outside"),
        ("saturation", "BAND_DB", [4.0, 2.0], "increasing"),
    ],
)
def test_invalid_values_are_reported(conf, section, key, value, message):
    conf[section][key] = value
    problems = config_problems(conf)
    assert len(problems) == 1 and message in problems[0]


def test_gate_check_levels_are_checked(conf):
    del conf["gate_check"]["levels"]["strict"]["n_qc"]
    conf["gate_check"]["levels"]["large"]["n_qcc"] = 1
    problems = config_problems(conf)
    assert "[gate_check.levels.strict] lacks n_qc" in problems
    assert "[gate_check.levels.large] : unknown n_qcc" in problems
    conf["gate_check"]["levels"] = {}
    assert "at least one level" in config_problems(conf)[0]
