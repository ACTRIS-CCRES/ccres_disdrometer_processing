import logging
import logging.config
import warnings
from enum import Enum, auto

LOG_FORMAT = r"%(levelname)s: [%(asctime)s] %(name)s %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class LogLevels(Enum):
    NOTSET = auto()
    DEBUG = auto()
    INFO = auto()
    CRITICAL = auto()
    WARNING = auto()
    ERROR = auto()

    @classmethod
    def get_by_verbosity_count(cls, verbosity: int):
        if verbosity == 0:
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            return cls.CRITICAL
        elif verbosity == 1:
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            return cls.INFO
        elif verbosity >= 2:
            return cls.DEBUG


def get_log_level_from_count(count: int) -> LogLevels:
    """get_log_level_from_count Dispatch the verbose count to Loglevels enum.

    Parameters
    ----------
    count : int
        Verbose count from CLI

    Returns
    -------
    LogLevels
        Corresponding Enum from the count

    """
    level = LogLevels.ERROR
    if count == 1:
        level = LogLevels.INFO
    if count >= 2:
        level = LogLevels.DEBUG
    return level


def add_stream_logging(logger: logging.Logger, level: LogLevels) -> logging.Logger:
    log_level = logging.getLevelName(level.name)
    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter(LOG_FORMAT)
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)
    return logger


def init_logger(level: LogLevels) -> None:
    """init_logger Init the stream logger for the project.

    Parameters
    ----------
    level : LogLevels
        Enum corresponding to the level we want

    """
    log_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"simple": {"datefmt": LOG_DATE_FORMAT, "format": LOG_FORMAT}},
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level.name,
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console"]},
    }

    logger = logging.getLogger()
    logging.config.dictConfig(log_dict)

    return logger
