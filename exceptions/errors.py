import typing 

class MicrogradError(Exception):
    def __init__(self,message: str) -> None:
        if not isinstance(message,str):
            raise TypeError("message must be string")
        
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

