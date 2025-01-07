#%%
import dynamics as dyn
import sympy as sp
import numpy as np


# %%
# defining kinematic parameters for robot
dh = [[0, 0, 0.4, 0],
      [0, 0, 0.4, 0],
      [0, 0, 0.4, 0]]

joint_type = ['r', 'r', 'r']

link_masses = [1, 1, 1]

# defining three different centers of mass, one for each link
r_coms = [sp.Matrix(3, 1, [-0.2, 0, 0]), sp.Matrix(3, 1, [-0.2, 0, 0]), sp.Matrix(3, 1, [-0.2, 0, 0])]

link_inertias = [sp.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0.01]]),
                 sp.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0.01]]),
                 sp.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0.01]])]

# note that gravity is defined positively (because it represents energy). All signs
# for gravity vectors are taken care of in the E-L equation itself.
arm = dyn.SerialArmDyn(dh,
                       jt=joint_type,
                       mass=link_masses,
                       r_com=r_coms,
                       link_inertia=link_inertias,
                       grav = sp.Matrix(3,1, [0, 9.81, 0]))


# %% [markdown]
# # Comparing E-L equation output to the answer from RNE from HW 08:
# ## Start with calculating $\tau$, given $q, \dot{q}, \ddot{q}$ (this is HW 08, problem 3a):
# %%

# defining numerical versions to pass to E-L
pi = 3.14159
q = [pi/4.0, pi/4.0, pi/4.0]
qd = [pi/6.0, -pi/4.0, pi/3.0]
qdd = [-pi/6.0, pi/3.0, pi/6.0]

# making q, qd, and qdd into numpy arrays, it works better if we pass
# lists into our E-L functions, but then multiply with the correct
# column vectors. This is a little clunky.
q_vec = np.array(q).reshape(3,1)
qd_vec = np.array(qd).reshape(3,1)
qdd_vec = np.array(qdd).reshape(3,1)

# we could write a function that does this, but I want you to see it explicitly
tau_EL = (arm.M(q) @ qdd_vec) + (arm.C(q, qd) @ qd_vec) + arm.G(q)

print("tau for E-L is:")
sp.pprint(tau_EL)


# %% [markdown]
# # Comparison for HW 08 - Problem 3, part b) - Mass Matrix

M_EL = arm.M(q)
print("generalized mass matrix from E-L is:")
sp.pprint(M_EL)

# %% [markdown]
# # Comparison for HW 08 - Problem 3, part c) - Coriolis terms:
# %%

# calculating C using the E-L function
C_EL = arm.C(q, qd) @ qd_vec
print("Coriolis vector from E-L is:")
print(C_EL)
