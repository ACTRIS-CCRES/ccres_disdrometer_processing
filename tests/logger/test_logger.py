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
    init_logger(level=LogLevels.INFO)
    lgr = logging.getLogger("ccres_disdrometer_processing")
    assert logging.getLevelName(lgr.handlers[0].level) == "INFO"
    assert LOG_FORMAT in logging.getLevelName(lgr.handlers[0].formatter._fmt)
