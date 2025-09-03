# %% [markdown]
# # HW 1: Introduction and Setup
# * Strongly consider completing the tutorial here (unless you already have significant experience with VS Code and Python) - https://code.visualstudio.com/docs/python/python-tutorial
# * Make sure to have set up your Python environment and installation of Python libraries correctly (see "requirements.txt" file in this folder, and this link for more about environments in VS Code - https://code.visualstudio.com/docs/python/environments#_creating-environments).
# * Run this notebook and check the output
# * If things are set up correctly, you should see two coordinate frames appear in a new window.
# * Play with these values a little to start to get some intuition for what these matrices mean.

# %%
# this cell will fail if you haven't correctly installed the libraries in the "requiremnts.txt" file
import numpy as np
import time
from visualization import VizScene

Tw_to_frame1 = np.eye(4)
viz = VizScene()
viz.add_frame(np.eye(4), label='world', axes_label='w')
viz.add_frame(Tw_to_frame1, label='frame1', axes_label='1')

time_to_run = 10
refresh_rate = 60
t = 0
start = time.time()
while t < time_to_run:
    t = time.time() - start

    # you can play with omega and p to see how they affect the frame
    omega = np.pi/2
    R = np.array([[np.cos(omega*t), -np.sin(omega*t), 0],
                  [np.sin(omega*t), np.cos(omega*t), 0],
                  [0, 0, 1]])
    p = np.array([1, 0, 0])

    Tw_to_frame1[:3,:3] = R
    Tw_to_frame1[:3,-1] = p
    viz.update(As=[np.eye(4), Tw_to_frame1])

    viz.hold(1/refresh_rate)

viz.close_viz() # could use viz.hold() to keep it open until manually closed

# %%
