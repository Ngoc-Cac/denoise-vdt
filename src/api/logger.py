import os
import logging
import logging.config
import sys
from logging.handlers import RotatingFileHandler

FORMAT = (
    "[%(asctime)s] %(levelname)s:PID-%(process)d:%(threadName)s:%(name)s: "
    "%(message)s"
)

if not os.path.exists("logs"):
    os.mkdir("logs")


def get_logger(
    name,
    fmt=None,
    log_filename=None,
    log_level=None,
    rotating_file_handler=False,
):
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
    })
    logger = logging.getLogger(name)
    logger.setLevel(log_level if log_level else logging.INFO)

    if not logger.handlers:
        log_formatter = logging.Formatter(fmt if fmt else FORMAT)

        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(log_formatter)
        logger.addHandler(stream_handler)

        if log_filename:
            if rotating_file_handler:
                file_handler = RotatingFileHandler(
                    "logs/api-server.log",
                    maxBytes=20_000_000,
                    backupCount=10,
                    encoding="utf-8"
                )
            else:
                file_handler = logging.FileHandler(
                    f"logs/{log_filename}", encoding="utf-8"
                )

            file_handler.setFormatter(log_formatter)
            logger.addHandler(file_handler)

    return logger
