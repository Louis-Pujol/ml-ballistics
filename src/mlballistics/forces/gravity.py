from .base_force import Force
from ..utils import _ei

import torch


class Gravity(Force):
    def __init__(self, g: float = 9.81) -> None:
        super().__init__()
        self._g = g
        return

    def __call__(self, state=None, obj=None) -> torch.Tensor:
        m = obj.mass
        return - m * self._g * _ei(3, 2)

    @property
    def g(self) -> float:
        return self._g

    @g.setter
    def g(self, value: float) -> None:
        self._g = value
