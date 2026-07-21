import typing 

class MicrogradError(Exception):
    def __init__(self,message: str) -> None:
        if not isinstance(message,str):
            raise TypeError("message must be a string")
        
        if not message.strip():
            raise ValueError("message cannot be empty")
        
        self.message = message

        super().__init__(message)

    def __str__(self) -> str:
        message = self.message
        return message
    
class ConfigurationError(MicrogradError):
    """Raised when project configuration is invalid."""


class DatasetValidationError(MicrogradError):
    """Raised when dataset validation fails."""


class ShapeMismatchError(MicrogradError):
    """Raised when tensor/vector dimensions are incompatible."""


class GraphCycleError(MicrogradError):
    """Raised when a computational graph contains a cycle."""


class GradientComputationError(MicrogradError):
    """Raised when gradient calculation becomes invalid."""


class CheckpointError(MicrogradError):
    """Raised when checkpoint save/load fails."""


class LoggingConfigurationError(MicrogradError):
    """Raised when logging cannot be configured."""


class VisualizationError(MicrogradError):
    """Raised when graph or plotting fails."""


class NumericalInstabilityError(MicrogradError):
    """Raised when NaN or Inf values appear."""


class InvalidActivationError(MicrogradError):
    """Raised for unsupported activation functions."""


class InvalidParameterError(MicrogradError):
    """Raised when model parameters are invalid."""


class ModelInitializationError(MicrogradError):
    """Raised when model creation fails."""


class TrainingError(MicrogradError):
    """Raised when the training loop cannot continue."""


class GradientCheckFailedError(MicrogradError):
    """Raised when analytical and numerical gradients disagree."""


class SerializationError(MicrogradError):
    """Raised when saving/loading serialized objects fails."""


class RandomSeedError(MicrogradError):
    """Raised when reproducibility setup fails."""