import numpy as np

# these DH parameters are based on solutions from HW 3, if you
# pick a different set that still describe the robots accurately,
# that's great.
a_len = 0.5
d_len = 0.35

dh_part_a = [[0, d_len, 0., np.pi/2.0],
            [0, 0, a_len, 0], 
            [0, 0, a_len, 0]]

dh_part_b = [[0, d_len, 0., -np.pi/2.0],
            [0, 0, a_len, 0], 
            [np.pi/2.0, 0, 0, np.pi/2.0], 
            [np.pi/2.0, d_len*2, 0, -np.pi/2.0],
            [0, 0, 0, np.pi/2],
            [0, d_len*2, 0, 0]]