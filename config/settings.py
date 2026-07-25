from __future__ import annotations

import copy
import json
from pathlib import Path

from ..exceptions.errors import ConfigurationError

class ModelConfig:
    def __init__(self, input_size: int, hidden_layers: list[int], output_size: int, activation: str) -> None:
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation = activation

        self.validate()


    def validate(self) -> None:
        if not isinstance(self.input_size, int):
            raise TypeError("input_size must be an int")
        
        if not isinstance(self.hidden_layers, list):
            raise TypeError("hidden_layers must be a list")
        
        if not isinstance(self.output_size, int):
            raise TypeError("output_size must be an int")

        if not isinstance(self.activation, str):
            raise TypeError("activation must be a str")

        if self.input_size < 1:
            raise ConfigurationError("input_size must be greater than 0")

        if not self.hidden_layers:
            raise ConfigurationError("hidden_layers list cannot be empty")

        for layer_size in self.hidden_layers:
            if not isinstance(layer_size, int):
                raise TypeError("each hidden_layer must be an int")

            if layer_size < 1:
                raise ConfigurationError("each hidden_layer must be greater than 0")

        if self.output_size < 1:
            raise ConfigurationError("output_size must be greater than 0")

        if not self.activation or self.activation not in ("relu","tanh"):
            raise ConfigurationError("activation must be one of: 'relu', 'tanh'")


    def to_dict(self) -> dict[str, object]:
        configuration = {
            "input_size": self.input_size,
            "hidden_layers": self.hidden_layers,
            "output_size": self.output_size,
            "activation": self.activation,
        }

        return configuration


class TrainingConfig:
    def __init__(self, learning_rate: float , epochs: int, batch_size: int, gradient_clip: float | None, random_seed: int) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.gradient_clip = gradient_clip
        self.random_seed = random_seed

        self.validate()

    def validate(self) -> None:
        if not isinstance(self.learning_rate, float):
            raise TypeError("learning_rate must be a float")

        if not isinstance(self.epochs, int):
                    raise TypeError("epochs must be a int")
        
        if not isinstance(self.batch_size, int):
                    raise TypeError("batch_size must be a int")
        
        if not isinstance(self.random_seed, int):
                    raise TypeError("random_seed must be a int")

        if self.learning_rate <= 0.0 or self.learning_rate > 1.0:
            raise ConfigurationError("learning_rate must be greater than 0.0 and less than or equal to 1.0")

        if self.epochs < 1:
            raise ConfigurationError(
                "epochs must be greater than 0."
            )

        if self.batch_size < 1:
            raise ConfigurationError(
            "batch_size must be greater than 0."
        )

        if self.gradient_clip is not None:
            if self.gradient_clip < 1:
                raise ConfigurationError(
                    "gradient_clip must be greater than 0."
                )

        if self.random_seed < 0:
            raise ConfigurationError(
                "random_seed must be greater than or equal to 0"
            )

    def to_dict(self) -> dict[str, object]:
        configuration = {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "gradient_clip": self.gradient_clip,
            "random_seed": self.random_seed,
        }

        return configuration


class VisualizationConfig:

    _SUPPORTED_GRAPH_FORMATS: frozenset[str] = frozenset(
        {"png", "svg", "pdf"}
    )

    def __init__(
        self,
        graph_format: str,
        figure_dpi: int,
        animation_fps: int,
    ) -> None:

        self.graph_format = graph_format
        self.figure_dpi = figure_dpi
        self.animation_fps = animation_fps

        self.validate()

    def validate(self) -> None:
        if not isinstance(self.graph_format, str):
            raise TypeError("graph_format must be a str.")

        if not isinstance(self.figure_dpi, int):
            raise TypeError("figure_dpi must be an int.")

        if not isinstance(self.animation_fps, int):
            raise TypeError("animation_fps must be an int.")

        graph_format = self.graph_format.lower()

        if graph_format not in self._SUPPORTED_GRAPH_FORMATS:
            supported_formats = ", ".join(
                sorted(self._SUPPORTED_GRAPH_FORMATS)
            )

            raise ConfigurationError(
                f"graph_format must be one of: {supported_formats}."
            )

        if self.figure_dpi <= 0:
            raise ConfigurationError(
                "figure_dpi must be greater than 0."
            )

        if self.animation_fps <= 0:
            raise ConfigurationError(
                "animation_fps must be greater than 0."
            )

        self.graph_format = graph_format

    def to_dict(self) -> dict[str, object]:
        configuration: dict[str, object] = {
            "graph_format": self.graph_format,
            "figure_dpi": self.figure_dpi,
            "animation_fps": self.animation_fps,
        }

        return configuration
    

class ProjectConfig:
    VERSION = 1

    def __init__(
        self,
        model: ModelConfig,
        training: TrainingConfig,
        visualization: VisualizationConfig,
    ) -> None:
        if not isinstance(model, ModelConfig):
            raise TypeError("model must be a ModelConfig instance.")

        if not isinstance(training, TrainingConfig):
            raise TypeError("training must be a TrainingConfig instance.")

        if not isinstance(visualization, VisualizationConfig):
            raise TypeError(
                "visualization must be a VisualizationConfig instance."
            )

        self.model = model
        self.training = training
        self.visualization = visualization

        self.validate()

    def load(self, path: str) -> None:
        if not isinstance(path, str):
            raise TypeError("path must be a str.")

        file_path = Path(path)

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Configuration file '{file_path}' does not exist."
            )

        try:
            with file_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as error:
            raise ConfigurationError(
                f"Invalid JSON configuration: {error}"
            ) from error

        required_sections = {
            "model",
            "training",
            "visualization",
        }

        missing = required_sections - data.keys()

        if missing:
            raise ConfigurationError(
                f"Missing configuration sections: {', '.join(sorted(missing))}."
            )

        try:
            self.model = ModelConfig(**data["model"])
            self.training = TrainingConfig(**data["training"])
            self.visualization = VisualizationConfig(
                **data["visualization"]
            )
        except TypeError as error:
            raise ConfigurationError(
                f"Invalid configuration structure: {error}"
            ) from error

        self.validate()

    def save(self, path: str) -> None:
        if not isinstance(path, str):
            raise TypeError("path must be a str.")

        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": self.VERSION,
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "visualization": self.visualization.to_dict(),
        }

        with file_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

    def validate(self) -> None:
        self.model.validate()
        self.training.validate()
        self.visualization.validate()
