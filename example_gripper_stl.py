"""
Example: Using STL gripper with robot arm visualization

This example shows how to add a custom STL gripper to a robot arm visualization.
"""

import numpy as np
import kinematics as kin
from visualization import VizScene

# Define a simple 3-DOF robot arm
dh = [[0, 0, 0.3, np.pi/2.0],
      [0, 0, 0.3, 0.0],
      [0, 0, 0.2, 0.0]]

joint_types = ['r', 'r', 'r']

# Create the arm
arm = kin.SerialArm(dh, jt=joint_types)

# Create visualization
viz = VizScene()

# Example 1: Add arm with default end effector (no STL)
viz.add_arm(arm, draw_frames=True)

# Example 2: Add arm with STL gripper
# Uncomment and modify path to your STL file:
# viz.add_arm(arm, draw_frames=True, gripper_stl='path/to/your/gripper.stl')

# Example 3: Add arm with STL gripper and custom offset
# The gripper_offset allows you to reorient the gripper relative to the tip frame
# gripper_offset = np.array([[1, 0, 0, 0],      # No rotation
#                            [0, 1, 0, 0],
#                            [0, 0, 1, 0.05],   # Offset 5cm in z
#                            [0, 0, 0, 1]])
# viz.add_arm(arm, draw_frames=True, 
#            gripper_stl='path/to/your/gripper.stl',
#            gripper_offset=gripper_offset)

# Example 4: Gripper rotated 90 degrees about z-axis
# import transforms as tr
# gripper_offset = tr.se3(tr.rotz(np.pi/2), [0, 0, 0.05])
# viz.add_arm(arm, draw_frames=True,
#            gripper_stl='path/to/your/gripper.stl',
#            gripper_offset=gripper_offset)

# Set initial joint configuration
q = [0, np.pi/4, np.pi/4]
viz.update(qs=[q])

# Display
viz.hold()
