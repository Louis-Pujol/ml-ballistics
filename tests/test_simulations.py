import torch

from mlballistics.objects import Sphere
from mlballistics.forces import Gravity, Drag
from mlballistics.scene import Scene


def test_basic_simulation():

    target = Sphere(
        radius=0.05,
        mass=1.0,
        initial_position=torch.Tensor([0.8, 0.8, 0.0]),
        initial_velocity=torch.Tensor([0.0, 0.0, 0.0]),
        force=None,
    )

    missile = Sphere(
        radius=0.05,
        mass=1.0,
        initial_position=torch.Tensor([0.0, 0.0, 0.0]),
        initial_velocity=torch.Tensor([1.0, 1.0, 0.0]),
        force=Gravity(),
    )

    scene = Scene(objects=[target, missile])
    scene.simulate(stop_time=1.0, n_steps=100)

    # Assert that the target do not moves
    assert torch.allclose(target.initial_position, target.trajectory)
    # Assert that the missile moves
    assert not torch.allclose(missile.initial_position, missile.trajectory)


def test_simulation_with_drag():

    initial_position = torch.Tensor([0.0, 0.0, 0.0])
    initial_velocity = torch.Tensor([1.0, 0.0, 1.0])

    missile_nodrag = Sphere(
        radius=0.05,
        mass=1.0,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        force=Gravity(),
    )

    missile_drag = Sphere(
        radius=0.05,
        mass=1.0,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        force=Gravity() + Drag(),
    )

    scene = Scene(objects=[missile_nodrag, missile_drag])
    scene.simulate(stop_time=1.0, n_steps=100)

    # assert that the missile with drag has a shorter range
    # than the missile without drag
    x_trajectory_nodrag = missile_nodrag.trajectory[:, 0]
    x_trajectory_drag = missile_drag.trajectory[:, 0]
    assert torch.all(x_trajectory_drag <= x_trajectory_nodrag)
