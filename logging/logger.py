import logging
import logging.handlers
import pathlib
import json
import typing
from datetime import datetime

from exceptions.errors import LoggingConfigurationError


class ProjectLogger:
    def __init__(
            self, 
            log_directory: str = "logs", 
            logger_name: str = "micrograd", 
            console_logging: bool = True, 
            file_logging: bool = True
            ) -> None:
        

        if not isinstance(log_directory, str):
            raise TypeError("log_directory must be a string")
        
        log_directory = log_directory.strip()

        if not log_directory:
            raise ValueError("log_directory cannot be empty")

        if not isinstance(logger_name, str):
            raise TypeError("logger_name must be a string")
        
        logger_name = logger_name.strip()

        if not logger_name:
            raise ValueError("logger_name cannot be empty")
        
        if not isinstance(console_logging, bool):
            raise TypeError("console_logging must be a boolean")
        
        if not isinstance(file_logging, bool):
            raise TypeError("file_logging must be a boolean")
        
        
        self.log_directory = pathlib.Path(log_directory)
        
        try:
            self.log_directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise LoggingConfigurationError(
                "Failed to create log directory"
            ) from error

        self.logger_name = logger_name

        self.console_logging = console_logging
        self.file_logging = file_logging

        self.run_identifier = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.json_file = "run.json"
        self.text_file = "run.log"
        
        self.json_log_path = self.log_directory / self.json_file
        self.text_log_path = self.log_directory / self.text_file

        self.logger = logging.getLogger(self.logger_name)

    def configure(self) -> None:
        if self.logger.handlers:
            return

        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        text_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        try:
            if self.console_logging:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.DEBUG)
                console_handler.setFormatter(text_formatter)

                self.logger.addHandler(console_handler)

            if self.file_logging:
                text_handler = logging.handlers.RotatingFileHandler(
                    filename=self.text_log_path,
                    maxBytes=5 * 1024 * 1024,
                    backupCount=5,
                    encoding="utf-8",
                )
                
                text_handler.setLevel(logging.DEBUG)
                text_handler.setFormatter(text_formatter)

                # json_handler = logging.handlers.RotatingFileHandler(
                #     filename=self.json_log_path,
                #     maxBytes=5 * 1024 * 1024,
                #     backupCount=5,
                #     encoding="utf-8",
                # )

                # json_handler.setLevel(logging.DEBUG)
                # json_handler.setFormatter(formatter)

                self.logger.addHandler(text_handler)
                # self.logger.addHandler(json_handler)

        except OSError as error:
            raise LoggingConfigurationError(
                "Failed to configure logging"
            ) from error

        if not self.logger.handlers:
            raise LoggingConfigurationError(
                "Logger has no configured handlers"
            )

        self.logger.info(
            "Logger initialized successfully"
        )

    def debug(self, message: str) -> None:
        message = message.strip()

        if not isinstance(message, str):
            raise TypeError("message must be a string")

        if not message:
            raise ValueError("message cannot be empty")

        self.logger.debug(message)

    def info(self, message: str) -> None:
        message = message.strip()

        if not isinstance(message, str):
            raise TypeError("message must be a string")

        if not message:
            raise ValueError("message cannot be empty")

        self.logger.info(message)

    def warning(self, message: str) -> None:
        message = message.strip()

        if not isinstance(message, str):
            raise TypeError("message must be a string")

        if not message:
            raise ValueError("message cannot be empty")

        self.logger.warning(message)

    def error(self, message: str, exception: Exception | None = None) -> None:
        message = message.strip()

        if not isinstance(message, str):
            raise TypeError("message must be a string")

        if not message:
            raise ValueError("message cannot be empty")

        if exception is not None:
            self.logger.error(message,exc_info=exception)
        else:
            self.logger.error(message)

    def critical(self, message: str, exception: Exception | None = None) -> None:
        message = message.strip()

        if not isinstance(message, str):
            raise TypeError("message must be a string")

        if not message:
            raise ValueError("message cannot be empty")

        if exception is not None:
            self.logger.critical(message,exc_info=exception)
        else:
            self.logger.critical(message)

    def log_metrics(self, epoch: int, loss: float, learning_rate: float, gradient_norm: float) -> None:
        if not isinstance(epoch, int):
            raise TypeError("epoch must be an int")

        if not isinstance(loss, float):
            raise TypeError("loss must be a float")

        if not isinstance(learning_rate, float):
            raise TypeError("learning_rate must be a float")

        if not isinstance(gradient_norm, float):
            raise TypeError("gradient_norm must be a float")

        