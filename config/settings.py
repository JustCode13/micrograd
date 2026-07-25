from exceptions.errors import ConfigurationError

class ModelConfig:
    def __init__(self, input_size: int, hidden_layers: list[int], output_size: int, activation: str) -> None:

        if not isinstance(input_size, int):
            raise TypeError("input_size must be an int")
        
        if not isinstance(hidden_layers, list):
            raise TypeError("hidden_layers must be a list")
        
        if not isinstance(output_size, int):
            raise TypeError("output_size must be an int")

        if not isinstance(activation, str):
            raise TypeError("activation must be a str")

        if input_size < 1:
            raise ConfigurationError("input_size must be greater than 0")

        if not hidden_layers:
            raise ConfigurationError("hidden_layers list cannot be empty")

        for layer_size in hidden_layers:
            if not isinstance(layer_size, int):
                raise TypeError("each hidden_layer must be an int")

            if layer_size < 1:
                raise ConfigurationError("each hidden_layer must be greater than 0")

        if output_size < 1:
            raise ConfigurationError("output_size must be greater than 0")

        activation = activation.strip()

        if not activation or activation not in ("relu","tanh"):
            raise ConfigurationError("activation must be one of: 'relu', 'tanh'")

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation = activation

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


class TraningConfig:
    def __init__(self, learning_rate: float , epochs: int, batch_size: int, gradient_clip: float | None, random_seed: int) -> None:

        if not isinstance(learning_rate, float):
            raise TypeError("learning_rate must be a float")

        if not isinstance(epochs, int):
                    raise TypeError("epochs must be a int")
        
        if not isinstance(batch_size, int):
                    raise TypeError("batch_size must be a int")
        
        if not isinstance(random_seed, int):
                    raise TypeError("random_seed must be a int")

        if learning_rate < 1 or learning_rate > 1:
            raise ConfigurationError("learning_rate must be greater than 0.0 and less than or equal to 1.0")

        if epochs < 1:
            raise ConfigurationError(
                "epochs must be greater than 0."
            )

        if batch_size < 1:
            raise ConfigurationError(
            "batch_size must be greater than 0."
        )

        if gradient_clip is not None:
            if gradient_clip < 1:
                raise ConfigurationError(
                    "gradient_clip must be greater than 0."
                )

        if random_seed < 1:
            raise ConfigurationError(
                "random_seed must be greater than or equal to 0"
            )

        