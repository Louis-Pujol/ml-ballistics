import torch
import pyvista as pv

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

        self._radius = radius

    def actor(self, time: int, **kwargs):

        if time == 0:
            position_np = self.initial_position.detach().cpu().numpy()
        else:
            position_np = self.trajectory[time].detach().cpu().numpy()

        prop = pv.Property(**kwargs)
        sphere = pv.Sphere(center=position_np, radius=self.radius)
        return super()._actor_from_mesh(mesh=sphere, prop=prop)

    @property
    def radius(self) -> float:
        """Get the radius of the sphere.

        Returns
        -------
        float
            Radius of the sphere.
        """
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        """Set the radius of the sphere.

        Parameters
        ----------
        value
            Radius of the sphere.
        """
        self._radius = value
        self.sectional_area = 4 * torch.pi * value ** 2
