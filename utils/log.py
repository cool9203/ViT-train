#! python3
# coding: utf-8


import inspect
import logging
import logging.config
import logging.handlers
import warnings
from pathlib import Path
from typing import Optional

FORMAT_STD = "%(asctime)s %(levelname)s %(pathname)s.%(lineno)d %(message)s"
FORMAT_DEBUG = "%(levelname)s %(pathname)s.%(lineno)d %(message)s"
FORMAT_MESSAGE = "%(message)s"
FORMAT_TIME_MESSAGE = "%(asctime)s %(levelname)s %(message)s"
DATEFMT_STD = "%Y/%m/%d %I:%M:%S"


def get_logger(
    logger_name: str = "root",
    propagate: bool = False,
    log_level: Optional[str] = None,
    log_fmt: str = "STD",
    log_path: str = "./",
    file_name: Optional[str] = None,
    file_mode: str = "a",
    log_count: int = 7,
    max_bytes: int = 10,  # 10MB
) -> logging.Logger:
    """Get logging.logger from set params

    Args:
        logger_name (str, optional): set logger name. Defaults to "root".
        propagate (bool, optional): set logger propagate, if True, will propagate to root, else will not propagate.
        log_level (str, optional): set log show level, if None will set "INFO". Defaults to None.
        log_fmt (str, optional): set show format, choices=["STD", "DEBUG", other]. Defaults to "STD".
        log_path (str, optional): set log save file path. Defaults to "./".
        file_name (str, optional): set log save file name. Defaults to None.
        file_mode (str, optional): set file mode, choices=["w", "a"]. Defaults to "w".

    Raises:
        Exception: _description_

    Returns:
        logging.Logger: logger, use .info(), .debug(), .error(), .warning() or .exception() to show message
    """
    log_level = "INFO" if log_level is None else log_level.upper()
    log_fmt = log_fmt.upper()

    if file_mode not in ["w", "a"]:
        raise Exception(f"error: argument file_mode: invalid choice: {file_mode} (choose from 'w', 'a')")

    try:
        log_fmt = globals().get(f"FORMAT_{log_fmt.upper()}")
    except KeyError:
        warnings.warn(f"Not have format: {log_fmt}", Warning)
        log_fmt = FORMAT_STD

    logger = logging.getLogger(logger_name)

    if len(logger.handlers) == 0:
        logger.setLevel(logging.getLevelName(log_level))
        formatter = logging.Formatter(log_fmt, datefmt=DATEFMT_STD)
        logger.propagate = propagate

        handler = logging.StreamHandler()
        handler.setLevel(logging.getLevelName(log_level))
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        if file_name is not None:
            Path(log_path).mkdir(parents=True, exist_ok=True)
            handler = logging.handlers.RotatingFileHandler(
                str(Path(log_path, f"{file_name}.log")),
                mode=file_mode,
                backupCount=log_count,
                encoding="utf8",
                maxBytes=max_bytes * 1024 * 1024,
                delay=True,
            )
            handler.setLevel(logging.getLevelName(log_level))
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        else:
            frm = inspect.stack()[1]
            mod = inspect.getmodule(frm[0])
            warnings.warn(f"{mod.__name__}: Not use log file", Warning)

    return logger


def main() -> None:
    pass


if __name__ == "__main__":
    main()
