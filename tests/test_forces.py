
import torch
from typing import Literal

from mlballistics.forces import *
from mlballistics.objects import Sphere

def _test_dependency(
        object,
        force,
        parameter,
        parameter_attachement: Literal["force", "object"],
        dependence: Literal["none", "linear", "quadratic"],
):
    """Test if a force has a certain dependence wrt a given parameter.

    Parameters
    ----------
    object
        The object on which the force is applied.
    force
        The force to test.
    parameter
        The name of the parameter to test.
    parameter_attachement
        Could be "force" or "object" depending on wether we test a parameter
        of the force or of the object.
    dependence
        Could be "none", "linear" or "quadratic" depending on the dependence
        of the force on the parameter we want to test.

    Raises
    ------
    AssertionError
        If the dependence is not verified.
    """
    
    object.force = force
    object.initial_position = torch.rand(3)
    object.initial_velocity = torch.rand(3)

    initial_force = object.forces_vector()

    coeff = torch.rand(1).item()

    if parameter_attachement == "force":
        setattr(
            object.force,
            parameter,
            getattr(force, parameter) * coeff
        )
    elif parameter_attachement == "object":
        setattr(
            object,
            parameter,
            getattr(object, parameter) * coeff
        )

    if dependence == "none":
        power = 0
    elif dependence == "linear":
        power = 1
    elif dependence == "quadratic":
        power = 2


    target = initial_force * coeff ** power

    if not torch.allclose(
        object.forces_vector(),
        target
    ):
         raise AssertionError(
                f"Error in test, relation between {parameter_attachement}.{parameter}"
                f" and {type(force)} is not {dependence}"
         )
         

def _test_force_dependencies(force, list_dependencies):
    """Test a list of dependencies for a force.

    Parameters
    ----------
    force
        The force to test.
    list_dependencies
        A list of parameters to test. Each parameter is a list of 3 elements
        with the following structure :
        [parameter, parameter_attachement, dependence].
    """

    object = Sphere(mass=1.0, radius=0.5)
    kwargs_names = ["parameter", "parameter_attachement", "dependence"]

    for params in list_dependencies:
        kwargs = dict(zip(kwargs_names, params))
        kwargs["force"] = force
        _test_dependency(object, **kwargs)


dependencies_drag = [
    ["density", "force", "linear"],
    ["drag_coefficient", "object", "linear"],
    ["sectional_area", "object", "linear"],
    ["mass", "object", "none"],
    ["initial_velocity", "object", "quadratic"],
    ["initial_position", "object", "none"],
]


dependencies_gravity = [
    ["g", "force", "linear"],
    ["mass", "object", "linear"],
    ["sectional_area", "object", "none"],
    ["initial_velocity", "object", "none"],
    ["initial_position", "object", "none"],
]

# As null force is zero, dependencies are "none", "linear" and "quadratic"
# a the same time
dependencies_null = [
    ["mass", "object", "none"],
    ["mass", "object", "quadratic"],
    ["sectional_area", "object", "none"],
    ["initial_velocity", "object", "none"],
    ["initial_position", "object", "linear"],
]

def test_dependencies_gravity():
        _test_force_dependencies(Gravity(), dependencies_gravity)

def test_dependencies_drag():
        _test_force_dependencies(Drag(), dependencies_drag)

def test_dependencies_null():
        _test_force_dependencies(NullForce(), dependencies_null)


def test_gravity():

    m = 2.3
    # A ball (mass = m) subject to gravity
    ball = Sphere(
        mass = m,
        radius=0.5,
    )
    ball.force = Gravity()

    # Initial conditions : ball is at rest
    ball.initial_position = torch.tensor([0.0, 0.0, 0.0])
    ball.initial_velocity = torch.tensor([0.0, 0.0, 0.0])
    # the only force applied is gravity
    assert torch.allclose(
        ball.forces_vector(),
        torch.tensor([0.0, 0.0, -9.81 * m])
    )

def test_drag():

    ball = Sphere(
        mass = 1.0,
        radius=0.5,
    )
    ball.force = Drag()

    ball.initial_position = torch.tensor([0.0, 0.0, 0.0])
    v = torch.rand(3)
    ball.initial_velocity = v

    # the drag force must be in the opposite direction of the velocity
    # ie : <f, v> = - ||v|| * ||f||
    forces = ball.forces_vector()
    assert torch.allclose(
        (v * forces).sum(),
        - torch.norm(v) * torch.norm(forces)
    )

def test_sum_forces():
    
    mass = 1.0
    radius = 0.5

    intitial_position = torch.rand(3)
    initial_velocity = torch.rand(3)

    ball = Sphere(
         mass=mass,
         radius=radius,
         initial_position=intitial_position,
         initial_velocity=initial_velocity,
         force = NullForce() + Gravity() + Drag()
        )

    ball_g = Sphere(
            mass=mass,
            radius=radius,
            initial_position=intitial_position,
            initial_velocity=initial_velocity,
            force = Gravity()
        )
    
    ball_d = Sphere(
            mass=mass,
            radius=radius,
            initial_position=intitial_position,
            initial_velocity=initial_velocity,
            force = Drag()
        )
    
    assert torch.allclose(
        ball.forces_vector(),
        ball_g.forces_vector() + ball_d.forces_vector()
    )

    


     

