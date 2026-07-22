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
        
        log_directory = log_directory.strip()
        logger_name = logger_name.strip()

        if not isinstance(log_directory, str):
            raise TypeError("log_directory must be a string")
        
        if not log_directory:
            raise ValueError("log_directory cannot be empty")

        if not isinstance(logger_name, str):
            raise TypeError("logger_name must be a string")
        
        if not logger_name:
            raise ValueError("logger_name cannot be empty")
        
        if not isinstance(console_logging, bool):
            raise TypeError("console_logging must be a boolean")
        
        if not isinstance(file_logging, bool):
            raise TypeError("file_logging must be a boolean")
        
        
        self.log_directory = pathlib.Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)

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

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        try:
            if self.console_logging:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.DEBUG)
                console_handler.setFormatter(formatter)

                self.logger.addHandler(console_handler)

            if self.file_logging:
                text_handler = logging.handlers.RotatingFileHandler(
                    filename=self.text_log_path,
                    maxBytes=5 * 1024 * 1024,
                    backupCount=5,
                    encoding="utf-8",
                )
                
                text_handler.setLevel(logging.DEBUG)
                text_handler.setFormatter(formatter)

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

        
        
