"""Parent class for all forces."""
import torch


class Force:
    """Abstract class for forces.

    A force is represented by a function that takes a state and an object as
    input and returns a force vector.
    """

    def __init__(self) -> None:
        pass

    def __add__(self, other):
        """Add two forces.

        Parameters
        ----------
        other
            Another force.

        Returns
        -------
            _descr
        """
        return SumForce(f1=self, f2=other)


class SumForce(Force):
    def __init__(self, f1: Force, f2: Force) -> None:
        super().__init__()
        self._f1 = f1
        self._f2 = f2

    def __call__(self, state=None, obj=None) -> torch.Tensor:
        return self._f1(
            state=state,
            obj=obj
        ) + self._f2(state=state, obj=obj)
