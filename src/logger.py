import logging

from src.config import logger


class Logger(logging.Logger):
    def __init__(self, name: str, level=None) -> None:

        if level is None:
            level = logger.level

        super().__init__(name, level=level)

        formatter = logging.Formatter(
            fmt=logger.fmt,
            datefmt=logger.datefmt,
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.setLevel(logger.level)
        self.addHandler(handler)
