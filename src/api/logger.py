import os
import logging
import logging.config
import sys
from logging.handlers import RotatingFileHandler


FORMAT = (
    "[%(asctime)s] %(levelname)s:PID-%(process)d:%(threadName)s:%(name)s: "
    "%(message)s"
)


def setup_rotating_handler(
    file,
    fmt=None,
    level=logging.DEBUG,
    *,
    maxBytes=20_000_000,
    backupCount=10,
    encoding="utf-8"
):
    formatter = logging.Formatter(fmt if fmt else FORMAT)
    handler = RotatingFileHandler(
        file,
        maxBytes=maxBytes,
        backupCount=backupCount,
        encoding=encoding
    )
    handler.setFormatter(formatter)
    handler.setLevel(level)
    return handler


def get_logger(name, fmt=None, log_dir='logs'):
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
    })
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        log_formatter = logging.Formatter(fmt if fmt else FORMAT)

        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(log_formatter)
        stream_handler.setLevel(logging.DEBUG)
        logger.addHandler(stream_handler)

        if log_dir:
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)

            info_handler = setup_rotating_handler(
                f"{log_dir}/api-server.log",
                fmt
            )
            logger.addHandler(info_handler)

    return logger
