# -*- coding: utf-8 -*-
"""
Impact of drag on a missile
===========================

This example shows the trajectory of a missile with and without drag.
"""

import torch
from mlballistics.objects import Sphere
from mlballistics.forces import Gravity, Drag
from mlballistics.scene import Scene

# %%
# Same initial conditions for both missiles.

initial_position = torch.Tensor([0.0, 0.0, 0.0])
initial_velocity = torch.Tensor([5.0, 0.0, 9.0])

# %%
# Missile without drag.

missile_nodrag = Sphere(
    radius=0.1,
    mass=1.0,
    initial_position=initial_position,
    initial_velocity=initial_velocity,
    force=Gravity(),
)

# %%
# Missile with drag.

missile_drag = Sphere(
    radius=0.1,
    mass=1.0,
    initial_position=initial_position,
    initial_velocity=initial_velocity,
    force=Gravity() + Drag(),
)

# %%
# Simulate the scene.

scene = Scene(objects=[missile_nodrag, missile_drag])
scene.simulate(stop_time=1.6, n_steps=100)

# %%
# Plot the trajectories.

import pyvista as pv

cpos = [(4.000001950189471, -12.056954193477605, 2.0641854759305716),
 (4.000001950189471, 0.0, 2.0641854759305716),
 (0.0, 0.0, 1.0)]

plotter_gif = pv.Plotter(notebook=False, off_screen=True)

plotter_gif.open_gif("missile.gif", fps=30)
for time in range(100):

    plotter_gif.clear_actors()

    actor1 = missile_drag.actor(time=time, color='red', opacity=0.5)
    actor2 = missile_nodrag.actor(time=time, color='blue', opacity=0.5)

    plotter_gif.add_actor(actor1)
    plotter_gif.add_actor(actor2)

    plotter_gif.camera_position = cpos
    plotter_gif.show_axes()
    plotter_gif.write_frame()

plotter_gif.show()
