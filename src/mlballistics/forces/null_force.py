import torch

from .base_force import Force


class NullForce(Force):
    def __init__(self) -> None:
        super().__init__()
        return

    def __call__(self, state=None, obj=None) -> torch.Tensor:
        return torch.zeros(3, dtype=torch.float)
