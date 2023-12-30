"""Scene that contains all objects and simulate the evolution."""
import torch

from ..objects import Object


class Scene:

    def __init__(self, objects: list[Object]):
        """Initialize a scene.

        Parameters
        ----------
        objects
            The list of objects in the scene.
        """
        self.objects = objects

    def simulate(
            self,
            stop_time: float = 1.0,
            n_steps: int = 100,
            ):
        """Simulate the scene.

        Parameters
        ----------
        time_step
            The time step between two iterations of the simulation.
        time
            The time of the simulation.
        """
        time = torch.linspace(0, stop_time, n_steps)

        for obj in self.objects:
            obj.simulate(time)
