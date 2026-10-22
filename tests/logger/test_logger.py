import logging

from ccres_disdrometer_processing.logger import (
    LOG_FORMAT,
    LogLevels,
    get_log_level_from_count,
    init_logger,
)


def test_log_level_from_count():
    assert get_log_level_from_count(0) == LogLevels.ERROR
    assert get_log_level_from_count(1) == LogLevels.INFO
    assert get_log_level_from_count(2) == LogLevels.DEBUG
    assert get_log_level_from_count(1000) == LogLevels.DEBUG


def test_init_logger():
    lgr = init_logger(LogLevels.INFO)

    assert logging.getLevelName(lgr.root.handlers[0].level) == "INFO"
    assert LOG_FORMAT in logging.getLevelName(lgr.root.handlers[0].formatter._fmt)


def test_log_level_by_verbosity_count():
    assert LogLevels.get_by_verbosity_count(0) == LogLevels.CRITICAL
    assert LogLevels.get_by_verbosity_count(1) == LogLevels.INFO
    assert LogLevels.get_by_verbosity_count(2) == LogLevels.DEBUG
    assert LogLevels.get_by_verbosity_count(5) == LogLevels.DEBUG
