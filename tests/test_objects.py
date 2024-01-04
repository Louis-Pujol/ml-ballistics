import torch
from mlballistics.objects import Sphere


def test_sphere():
    """Test the Sphere object."""

    sphere = Sphere(radius=1)

    assert sphere.radius == 1
    assert sphere.sectional_area == 4 * torch.pi * 1 ** 2

    # Change the radius
    sphere.radius = 2
    assert sphere.radius == 2
    assert sphere.sectional_area == 4 * torch.pi * 2 ** 2
