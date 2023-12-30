import torch
from torchdiffeq import odeint
from typing import Optional


from ..forces import NullForce, Force


class Object:
    """Base class for all objects in the simulation.

    Parameters
    ----------
    mass
        Mass of the object.

    """
    def __init__(
            self,
            mass: float = 1.0,
            drag_coefficient: float = 0.0,
            sectional_area: float = 1.0,
            initial_position: Optional[torch.Tensor] = None,
            initial_velocity: Optional[torch.Tensor] = None,
            force: Optional[Force] = None,
            ) -> None:
        self._mass = mass
        self._drag_coefficient = drag_coefficient
        self._sectional_area = sectional_area
        if force is None:
            self._force = NullForce()
        else:
            self._force = force

        if initial_position is None:
            self._initial_position = torch.zeros(3, dtype=torch.float)
        else:
            self._initial_position = initial_position

        if initial_velocity is None:
            self._initial_velocity = torch.zeros(3, dtype=torch.float)
        else:
            self._initial_velocity = initial_velocity

    def ode_func(self, t, y):
        """ODE function for the object.

        Parameters
        ----------
        t
            Time.
        y
            State vector of the object.

        Returns
        -------
        torch.Tensor
            Derivative of the state vector.
        """
        return torch.cat([
            y[3:6],
            self.forces_vector(y) / self.mass,
            ]
        )

    def simulate(self, time: torch.Tensor):
        """Simulate the object.

        Parameters
        ----------
        time
            Time of the simulation.
        """
        self._states = odeint(
            self.ode_func,
            self.initial_state,
            t=time,
            method="rk4",
        )

        self._trajectory = self._states[:, :3]

    def forces_vector(self, state=None) -> torch.Tensor:
        """Vector of forces for the object.

        Parameters
        ----------
        state
            State vector of the object.

        Returns
        -------
        torch.Tensor
            Force acting on the object.
        """
        state = self.initial_state if state is None else state
        return self._force(state, self)

    @property
    def initial_state(self) -> torch.Tensor:
        """Initial state vector of the object.

        Parameters
        ----------
        position
            Position of the object.
        velocity
            Velocity of the object.

        Returns
        -------
        torch.Tensor
            State vector of the object.
        """
        return torch.cat([
            self.initial_position,
            self.initial_velocity,
            ])

    @property
    def mass(self) -> float:
        """Get the mass of the object.

        Returns
        -------
        float
            Mass of the object.
        """
        return self._mass

    @mass.setter
    def mass(self, value: float) -> None:
        """Set the mass of the object.

        Parameters
        ----------
        value
            Mass of the object.
        """
        self._mass = value

    @property
    def drag_coefficient(self) -> float:
        """Get the drag coefficient of the object.

        Returns
        -------
        float
            Drag coefficient of the object.
        """
        return self._drag_coefficient

    @drag_coefficient.setter
    def drag_coefficient(self, value: float) -> None:
        """Set the drag coefficient of the object.

        Parameters
        ----------
        value
            Drag coefficient of the object.
        """
        self._drag_coefficient = value

    @property
    def sectional_area(self) -> float:
        """Get the sectional area of the object.

        Returns
        -------
        float
            Sectional area of the object.
        """
        return self._sectional_area

    @sectional_area.setter
    def sectional_area(self, value: float) -> None:
        """Set the sectional area of the object.

        Parameters
        ----------
        value
            Sectional area of the object.
        """
        self._sectional_area = value

    @property
    def force(self) -> Force:
        """Get the force acting on the object.

        Returns
        -------
        Force
            Force acting on the object.
        """
        return self._force

    @force.setter
    def force(self, value: Optional[Force]) -> None:
        """Set the force acting on the object.

        Parameters
        ----------
        value
            Force acting on the object.
        """
        if value is None:
            self._force = NullForce()
        else:
            self._force = value

    @property
    def initial_position(self) -> torch.Tensor:
        """Get the initial position of the object.

        Returns
        -------
        torch.Tensor
            Initial position of the object.
        """
        return self._initial_position

    @initial_position.setter
    def initial_position(self, value: torch.Tensor) -> None:
        """Set the initial position of the object.

        Parameters
        ----------
        value
            Initial position of the object.
        """
        self._initial_position = value

    @property
    def initial_velocity(self) -> torch.Tensor:
        """Get the initial velocity of the object.

        Returns
        -------
        torch.Tensor
            Initial velocity of the object.
        """
        return self._initial_velocity

    @initial_velocity.setter
    def initial_velocity(self, value: torch.Tensor) -> None:
        """Set the initial velocity of the object.

        Parameters
        ----------
        value
            Initial velocity of the object.
        """
        self._initial_velocity = value

    @property
    def trajectory(self) -> torch.Tensor:
        """Get the trajectory of the object.

        Returns
        -------
        torch.Tensor
            Trajectory of the object.
        """
        return self._trajectory
