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

Tw_to_frame1 = np.array([[1, 0,  0,  1],
                         [0, 0, -1,  0],
                         [0, 1,  0,  0],
                         [0, 0,  0,  1]])
                         
viz = VizScene()
viz.add_frame(np.eye(4), label='world')
viz.add_frame(Tw_to_frame1, label='frame1')

time_to_run = 10
refresh_rate = 60
for i in range(refresh_rate * time_to_run):
    t = time.time()
    Tw_to_frame1 = np.array([[np.cos(np.pi/2*t), -np.sin(np.pi/2*t), 0, 1],
                             [np.sin(np.pi/2*t), np.cos(np.pi/2*t), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 0]])
    viz.update(As=[np.eye(4), Tw_to_frame1])
    time.sleep(1.0/refresh_rate)
    
viz.close_viz()




# %%
