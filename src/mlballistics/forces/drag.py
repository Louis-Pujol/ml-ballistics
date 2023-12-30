import torch

from .base_force import Force


class Drag(Force):
    r"""Drag force.

    The drag force is also known as the air resistance or the fluid resistance.
    It is a force that opposes the motion of an object through a fluid, it is
    directed in the opposite direction to the velocity of the object and it is
    proportional to the velocity of the object squared.

    The drag force is given by the following formula:
    $$ F = - C * ||v||^2 * \hat{v} $$
    where $C$ is the drag coefficient, $v$ is the velocity and $\hat{v}$ is the
    unit vector in the direction of $v$.

    The drag coefficient is given by the following formula:
    $$ C = \frac{\rho A c_d}{2} $$
    where

    - $\rho$ is the fluid density
    - $A$ is the cross-sectional area of the object
    - $c_d$ is the drag coefficient of the object

    See:
    https://en.wikipedia.org/wiki/Drag_(physics)
    https://en.wikipedia.org/wiki/Drag_coefficient



    Parameters
    ----------
    density
        Fluid density.

    """

    def __init__(
            self,
            density: float = 1.0,
            ) -> None:
        super().__init__()
        self._density = density

    def __call__(self, state=None, obj=None) -> torch.Tensor:
        v = state[3:6]
        v_norm = torch.norm(v)
        C = self._density * obj.drag_coefficient * obj.sectional_area / 2
        return - C * v_norm * v

    @property
    def density(self) -> float:
        return self._density

    @density.setter
    def density(self, value: float) -> None:
        self._density = value
