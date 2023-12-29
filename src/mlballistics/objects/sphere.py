import torch
from typing import Optional

from .base_object import Object
from ..constants import SPHERE_DRAG_COEFFICIENT
from ..forces import NullForce, Force

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
            sectional_area = 4 * torch.pi * radius ** 2,
            **kwargs,
            )

    