import torch

from .base_object import Object
from ..constants import SPHERE_DRAG_COEFFICIENT


class Sphere(Object):

    def __init__(
            self,
            radius: float = 1.0,
            mass: float = 1.0,
            **kwargs,
            ) -> None:

        super().__init__(
            mass=mass,
            drag_coefficient=SPHERE_DRAG_COEFFICIENT,
            sectional_area=4 * torch.pi * radius ** 2,
            **kwargs,
            )
