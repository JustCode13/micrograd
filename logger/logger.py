import logging
import logging.handlers
import pathlib
import json
import typing
from datetime import datetime, timezone

from exceptions.errors import LoggingConfigurationError

class JsonFormatter(logging.Formatter):

    def format(self, record):
        log = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        return json.dumps(log)

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

                json_handler = logging.handlers.RotatingFileHandler(
                    filename=self.json_log_path,
                    maxBytes=5 * 1024 * 1024,
                    backupCount=5,
                    encoding="utf-8",
                )

                json_handler.setLevel(logging.DEBUG)
                json_handler.setFormatter(JsonFormatter())

                self.logger.addHandler(text_handler)
                self.logger.addHandler(json_handler)

        except OSError as error:
            raise LoggingConfigurationError(
                "Failed to configure logging"
            ) from error

        if not self.logger.handlers:
            raise LoggingConfigurationError(
                "Logger has no configured handlers"
            )

        self.logger.info(
            f"Logger {self.logger_name} initialized successfully"
        )

    def debug(self, message: str) -> None:

        if not isinstance(message, str):
            raise TypeError("message must be a string")

        message = message.strip()

        if not message:
            raise ValueError("message cannot be empty")

        self.logger.debug(message)

    def info(self, message: str) -> None:

        if not isinstance(message, str):
            raise TypeError("message must be a string")

        message = message.strip()

        if not message:
            raise ValueError("message cannot be empty")

        self.logger.info(message)

    def warning(self, message: str) -> None:

        if not isinstance(message, str):
            raise TypeError("message must be a string")

        message = message.strip()

        if not message:
            raise ValueError("message cannot be empty")

        self.logger.warning(message)

    def error(self, message: str, exception: Exception | None = None) -> None:

        if not isinstance(message, str):
            raise TypeError("message must be a string")

        message = message.strip()

        if not message:
            raise ValueError("message cannot be empty")

        if exception is not None:
            self.logger.error(message,exc_info=exception)
        else:
            self.logger.error(message)

    def critical(self, message: str, exception: Exception | None = None) -> None:

        if not isinstance(message, str):
            raise TypeError("message must be a string")

        message = message.strip()

        if not message:
            raise ValueError("message cannot be empty")

        if exception is not None:
            self.logger.critical(message,exc_info=exception)
        else:
            self.logger.critical(message)

    def log_metrics(self, epoch: int, loss: float, learning_rate: float, gradient_norm: float) -> None:
        if not isinstance(epoch, int):
            raise TypeError("epoch must be an int")

        if (epoch < 0):
            raise ValueError("epoch must be greater than or equal 0")

        if not isinstance(loss, float):
            raise TypeError("loss must be a float")

        if (loss < 0):
            raise ValueError("loss must be greater than or equal 0")

        if not isinstance(learning_rate, float):
            raise TypeError("learning_rate must be a float")

        if (learning_rate <= 0):
            raise ValueError("learning_rate must be greater than 0")

        if not isinstance(gradient_norm, float):
            raise TypeError("gradient_norm must be a float")

        if (gradient_norm < 0):
            raise ValueError("gradient_norm must be greater than 0")

        metrics = {
            "epoch": epoch,
            "loss": loss,
            "learning_rate": learning_rate,
            "gradient_norm": gradient_norm,
        }

        timestamp = datetime.now().isoformat()

        metrics["timestamp"] = timestamp

        metrics["run_identifier"] = self.run_identifier

        json_metrics = json.dumps(metrics)

        self.logger.info(json_metrics)

        summary = (
            f"Epoch {epoch} | "
            f"Loss: {loss:.6f} | "
            f"LR: {learning_rate:.6f} | "
            f"Grad Norm: {gradient_norm:.6f}"
        )

        self.logger.info(summary)

    def log_configuration(self,configuration: dict[str,object],) -> None:
        if not isinstance(configuration, dict):
            raise TypeError("configuration must be a dictionary")

        timestamp = datetime.now(timezone.utc).isoformat()

        record = {
            "event": "configuration",
            "run_identifier": self.run_identifier,
            "timestamp": timestamp,
            "configuration": configuration,
        }
        try:
            json_record = json.dumps(record)
        except TypeError as error:
            raise LoggingConfigurationError("Failed to serialize configuration to JSON.") from error

        self.logger.info(json_record)

    def log_checkpoint(self, checkpoint_path: str, epoch: int) -> None:
        if not isinstance(checkpoint_path, str):
            raise TypeError("checkpoint_path must be a string")

        if not isinstance(epoch, int):
            raise TypeError("epoch must be an int")

        checkpoint_path = checkpoint_path.strip()

        if not checkpoint_path:
            raise ValueError("checkpoint_path cannot be empty")

        if epoch < 0:
            raise ValueError("epoch mmust be greater than or equal to 0")

        normalized_path = pathlib.Path(checkpoint_path)

        path_string = str(normalized_path)

        timestamp = datetime.now(timezone.utc).isoformat()

        record = {
            "event": "checkpoint_created",
            "run_identifier": self.run_identifier,
            "timestamp": timestamp,
            "checkpoint_path": path_string,
            "epoch": epoch,
        }

        try:
            json_record = json.dumps(record)
        except TypeError as error:
            raise LoggingConfigurationError(
                "Failed to serialize configuration to JSON" 
            ) from error

        self.logger.info(json_record)

    def log_dataset_statistics(self, statistics: dict[str, object]) -> None:
        if not isinstance(statistics, dict):
            raise TypeError("statistics must be a dictionary")

        timestamp = datetime.now(timezone.utc).isoformat()

        statistics["run_identifier"] = self.run_identifier
        statistics["timestamp"] = timestamp

        json_statistics = json.dumps(statistics)

        self.logger.info(json_statistics)

    def log_gradient_failure(self, parameter_name: str) -> None:

        if not isinstance(parameter_name, str):
            raise TypeError("parameter_name must be a str")

        parameter_name = parameter_name.strip()    

        if not parameter_name:
            raise ValueError("parameter_name cannot be empty")

        timestamp = datetime.now(timezone.utc).isoformat()

        record = {
            "event": "gradient_failure",
            "run_identifier": self.run_identifier,
            "timestamp": timestamp,
            "parameter_name": parameter_name,
        }

        try:
            json_record = json.dumps(record)
        except OSError as error:
            raise LoggingConfigurationError(
                "Failed to serialize parameter_name in JSON"
            ) from error

        self.error(json_record)

    def export_run_summary(self) -> dict[str, typing.Any]:

        summary = {
            "run_identifier": self.run_identifier,
            "logger_name": self.logger_name,
            "log_directory": str(self.log_directory),
            "text_log_path": str(self.text_log_path),
            "json_log_path": str(self.json_log_path)
        }

        return summary

    def close(self) -> None:
        for handler in self.logger.handlers:
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)
