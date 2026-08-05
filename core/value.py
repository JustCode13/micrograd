from __future__ import annotations

import math
import typing
import collections

from .graph import ComputationalGraph
from ..exceptions.errors import GraphCycleError, GradientComputationError


class Value:
    """
    Represents one scalar value inside the computational graph.

    Each Value object stores:
    - the actual scalar number
    - its gradient
    - references to previous Value nodes
    - the operation that created it
    - a local backward function for gradient propagation
    """

    def __init__(
        self,
        data: float,
        children: tuple["Value", ...] = (),
        operation: str = "",
        label: str | None = None,
    ) -> None:
        if not isinstance(data, (int, float)):
            raise TypeError(
                f"Value data must be int or float, got {type(data).__name__}"
            )

        data = float(data)

        if not math.isfinite(data):
            raise ValueError(
                f"Value data must be finite, got {data}"
            )

        self.data: float = data
        self.grad: float = 0.0

        self._prev: tuple["Value", ...] = children
        self._op: str = operation
        self.label: str | None = label

        self._backward: typing.Callable[[], None] = lambda: None

        self._graph: ComputationalGraph | None = None


    def _ensure_value(
        self,
        other: float | Value
    ) -> Value:
        if isinstance(other, Value):
            return other

        if isinstance(other, (int, float)):
            return Value(float(other))

        raise TypeError(
            f"Expected Value or numeric type, got {type(other).__name__}"
        )


    def __add__(
        self,
        other: float | Value
    ) -> Value:
        other = self._ensure_value(other)

        output = Value(
            self.data + other.data,
            children=(self, other),
            operation="+",
        )

        def _backward() -> None:
            self.grad += output.grad
            other.grad += output.grad

        output._backward = _backward

        ComputationalGraph.register_node(output)

        return output


    def __radd__(
        self,
        other: float | Value
    ) -> Value:
        return self.__add__(other)


    def __sub__(
        self,
        other: float | Value
    ) -> Value:
        other = self._ensure_value(other)

        output = Value(
            self.data - other.data,
            children=(self, other),
            operation="-",
        )

        def _backward() -> None:
            self.grad += output.grad
            other.grad -= output.grad

        output._backward = _backward

        ComputationalGraph.register_node(output)

        return output


    def __rsub__(
        self,
        other: float | Value
    ) -> Value:
        other = self._ensure_value(other)
        return other.__sub__(self)


    def __mul__(
        self,
        other: float | Value
    ) -> Value:
        other = self._ensure_value(other)

        output = Value(
            self.data * other.data,
            children=(self, other),
            operation="*",
        )

        def _backward() -> None:
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        output._backward = _backward

        ComputationalGraph.register_node(output)

        return output


    def __rmul__(
        self,
        other: float | Value
    ) -> Value:
        return self.__mul__(other)


    def __truediv__(
        self,
        other: float | Value
    ) -> Value:
        other = self._ensure_value(other)

        if other.data == 0:
            raise ZeroDivisionError(
                "Cannot divide Value by zero"
            )

        output = Value(
            self.data / other.data,
            children=(self, other),
            operation="/",
        )

        def _backward() -> None:
            self.grad += (1 / other.data) * output.grad
            other.grad += (
                -self.data /
                (other.data ** 2)
            ) * output.grad

        output._backward = _backward

        ComputationalGraph.register_node(output)

        return output


    def __rtruediv__(
        self,
        other: float | Value
    ) -> Value:
        other = self._ensure_value(other)
        return other.__truediv__(self)


    def __pow__(
        self,
        exponent: float | int
    ) -> Value:
        if not isinstance(exponent, (int, float)):
            raise TypeError(
                "Power exponent must be int or float"
            )

        if not math.isfinite(float(exponent)):
            raise ValueError(
                "Power exponent must be finite"
            )

        output = Value(
            self.data ** exponent,
            children=(self,),
            operation=f"**{exponent}",
        )

        def _backward() -> None:
            self.grad += (
                exponent *
                (self.data ** (exponent - 1))
            ) * output.grad

        output._backward = _backward

        ComputationalGraph.register_node(output)

        return output


    def __neg__(self) -> Value:
        return self * -1


    def tanh(self) -> Value:
        value = math.tanh(self.data)

        output = Value(
            value,
            children=(self,),
            operation="tanh",
        )

        def _backward() -> None:
            self.grad += (
                (1 - value ** 2) *
                output.grad
            )

        output._backward = _backward

        ComputationalGraph.register_node(output)

        return output


    def relu(self) -> Value:
        value = max(0.0, self.data)

        output = Value(
            value,
            children=(self,),
            operation="relu",
        )

        def _backward() -> None:
            self.grad += (
                (1.0 if self.data > 0 else 0.0)
                *
                output.grad
            )

        output._backward = _backward

        ComputationalGraph.register_node(output)

        return output


    def exp(self) -> Value:
        value = math.exp(self.data)

        output = Value(
            value,
            children=(self,),
            operation="exp",
        )

        def _backward() -> None:
            self.grad += value * output.grad

        output._backward = _backward

        ComputationalGraph.register_node(output)

        return output


    def log(self) -> Value:
        if self.data <= 0:
            raise ValueError(
                "Natural logarithm requires value greater than zero"
            )

        output = Value(
            math.log(self.data),
            children=(self,),
            operation="log",
        )

        def _backward() -> None:
            self.grad += (
                (1 / self.data)
                *
                output.grad
            )

        output._backward = _backward

        ComputationalGraph.register_node(output)

        return output

    def sigmoid(self) -> Value:
        """
        Sigmoid activation function.

        Formula:
            sigmoid(x) = 1 / (1 + e^-x)
        """

        if self.data >= 0:
            exp_value = math.exp(-self.data)
            result = 1 / (1 + exp_value)

        else:
            exp_value = math.exp(self.data)
            result = exp_value / (1 + exp_value)

        output = Value(
            result,
            children=(self,),
            operation="sigmoid",
        )

        def _backward() -> None:
            self.grad += (
                result *
                (1 - result)
                *
                output.grad
            )

        output._backward = _backward

        ComputationalGraph.register_node(output)

        return output


    def backward(self) -> None:
        """
        Perform reverse-mode automatic differentiation.

        Traverses the graph in reverse topological order and applies
        each node's local backward function.
        """

        try:
            nodes = ComputationalGraph.topological_sort(self)

        except GraphCycleError:
            raise

        if not nodes:
            raise GradientComputationError(
                "Cannot perform backward pass on an empty graph"
            )

        self.grad = 1.0

        for node in reversed(nodes):
            node.check_finite()

            try:
                node._backward()

            except Exception as error:
                raise GradientComputationError(
                    f"Gradient computation failed for operation '{node._op}': {error}"
                ) from error

        for node in nodes:
            node.check_finite()


    def zero_grad(self) -> None:
        """
        Reset accumulated gradient.
        """

        self.grad = 0.0


    def detach(self) -> Value:
        """
        Create a new Value disconnected from the graph.
        """

        detached = Value(
            data=self.data,
            label=self.label,
        )

        return detached


    def clone(self) -> Value:
        """
        Create a copy of this Value node.
        """

        cloned = Value(
            data=self.data,
            children=self._prev,
            operation=self._op,
            label=self.label,
        )

        cloned.grad = self.grad

        return cloned


    def check_finite(self) -> None:
        """
        Check whether data and gradients are finite.
        """

        if not math.isfinite(self.data):
            raise GradientComputationError(
                f"Non-finite value detected: data={self.data}"
            )

        if not math.isfinite(self.grad):
            raise GradientComputationError(
                f"Non-finite gradient detected: grad={self.grad}"
            )


    def visualize_name(self) -> str:
        """
        Generate readable name for graph visualization.
        """

        if self.label:
            return self.label

        if self._op:
            return f"Value({self.data:.4f}, op={self._op})"

        return f"Value({self.data:.4f})"


    def __repr__(self) -> str:
        """
        Developer-friendly representation.
        """

        return (
            f"Value("
            f"data={self.data}, "
            f"grad={self.grad}, "
            f"op='{self._op}'"
            f")"
        )
