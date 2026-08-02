import math
import typing
import collections


class Value:
    def __init__(
        self,
        data: float,
        children: tuple["Value", ...],
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
        self._backward = lambda: None
