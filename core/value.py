from __future__ import annotations
import math
import typing
import collections


class Value:
    def __init__(
        self,
        data: float,
        children: tuple[Value, ...] = (),
        operation: str = "",
        label: str | None = None,
    ) -> None:

        if not isinstance(data, (int, float)):
            raise TypeError("data must be int or float")

        if math.isnan(data):
            raise ValueError("data cannot be nan")

        if math.isinf(data):
            raise ValueError("data cannot be inf")

        self.data = data
        self.grad = 0.0
        self._prev = children
        self._op = operation
        self.label = label
        self.backward = lambda: None

    def __add__(self, other: float | Value) -> Value:
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(
            self.data + other.data,
            children=(self, other),
            operation="+",
        )

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out.backward = _backward

        # graph.register_node(out);

        return out

    def __radd__(self, other: float | Value) -> Value:
        return self.__add__(other)

    def __sub__(self, other: float | Value) -> Value:
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(
            self.data - other.data,
            children=(self, other),
            operation="-",
        )

        def _backward():
            self.grad += out.grad
            other.grad += -out.grad

        out.backward = _backward

        # graph.register_node(out)

        return out

    def __rsub__(self, other: float | Value) -> Value:
        return self.__sub__(other)

    def __mul__(self, other: float | Value) -> Value:
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(
            self.data * other.data,
            children=(self, other),
            operation="*",
        )

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out.backward = _backward

        # graph.register_node(out)

        return out

    def __rmul__(self, other: float | Value) -> Value:
        return self.__mul__(other)

    def __truediv__(self, other: float | Value) -> Value:
        if not isinstance(other, Value):
            other = Value(other)

        if other.data == 0:
            raise ZeroDivisionError("value must not be zero")

        out = Value(self.data / other.data, children=(self, other), operation="/")

        def _backward():
            self.grad += out.grad * (1 / other.data)
            other.grad += out.grad * (-self.data / other.data**2)

        self.backward = _backward

        # graph.register_node(out)

        return out

    def __rtruediv__(self, other):
        return self.__truediv__(other)
