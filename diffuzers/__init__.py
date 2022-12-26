import sys

from loguru import logger


logger.configure(handlers=[dict(sink=sys.stderr, format="> <level>{level:<7} {message}</level>")])

__version__ = "0.0.5"
