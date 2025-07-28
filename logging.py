from enum import Enum, auto
from datetime import datetime
import enum
from typing import Optional


class LogLevel(Enum):
    INFO = auto()
    DEBUG = auto()
    ERROR = auto()
    WARNING = auto()


class Verbosity(Enum):
    INFO = auto()
    DEBUG = auto()


log_level_map = ["INFO", "DEBUG", "ERROR", "WARNING"]


class Logging:
    def __init__(
        self,
        file_name,
        verbosity: Verbosity = Verbosity.INFO,
        user_name: Optional[str] = "myapp",
    ) -> None:
        self.name = user_name
        self.file_name = file_name
        self.verbos = verbosity

    def log(
        self,
        log_level: LogLevel,
        message: str,
        show_in_console: bool,
        *args,
        **keywords,
    ):
        pass

    def _formatting_message(
        self, log_level: LogLevel, message: str, *args, **keywords
    ) -> str:
        return ""

    def _write_to_console(self, message: str) -> None:
        pass

    def _write_to_file(self, message: str) -> None:
        pass

    def info(self, message: str, *args, **keywords) -> None:
        pass

    def debug(self, message: str, *args, **keywords) -> None:
        pass

    def warning(self, message: str, *args, **keywords) -> None:
        pass

    def error(self, message: str, *args, **keywords) -> None:
        pass

