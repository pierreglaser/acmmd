import logging
from typing import Literal
from typing_extensions import TypeAlias

LOG_LEVELS = {
    "calibration.kernels": logging.WARNING,
    "calibration.statistical_tests.cgof_mmd": logging.WARNING,
    "calibration.statistical_tests.skce": logging.WARNING,
    "calibration.statistical_tests.kccsd": logging.WARNING,
    "calibration.u_statistics": logging.WARNING,
}

LOG_LEVELS_KEYS_T: TypeAlias = Literal[
    "calibration.kernels",
    "calibration.statistical_tests.cgof_mmd",
    "calibration.statistical_tests.skce",
    "calibration.statistical_tests.kccsd",
    "calibration.u_statistics",
]


def set_log_level(name: LOG_LEVELS_KEYS_T, level: int) -> None:
    global LOG_LEVELS
    LOG_LEVELS[name] = level


def get_log_level(name: LOG_LEVELS_KEYS_T) -> int:
    return LOG_LEVELS[name]


# Copied from from sbi-benchmark: sbibm/utils/logging.py
def get_logger(name: LOG_LEVELS_KEYS_T, console_logging: bool = True) -> logging.Logger:
    """Gets logger with given name, while setting level and optionally adding handler

    Note: Logging to `sys.stdout` for Jupyter as done in this Gist
    https://gist.github.com/joshbode/58fac7ababc700f51e2a9ecdebe563ad

    Args:
        name: Name of logger
        level: Log level
        console_logging: Whether or not to log to console

    Returns:
        Logger
    """
    log = logging.getLogger(name)

    has_stream_handler = False
    for h in log.handlers:
        if type(h) == logging.StreamHandler:
            has_stream_handler = True
    if console_logging and not has_stream_handler:
        console_handler = logging.StreamHandler()
        log.addHandler(console_handler)

    log.setLevel(get_log_level(name))

    return log
