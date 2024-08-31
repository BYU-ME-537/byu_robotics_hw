"""
dynamics Module - Contains code for:
- Dynamic SerialArm class
- RNE Algorithm
- Euler - Lagrange formulation

John Morrell, Jan 28 2022
Tarnarmour@gmail.com

modified by: 
Marc Killpack, Nov. 4, 2022
"""

import numpy as np
from kinematics import SerialArm
from utility import skew

eye = np.eye(4)

class SerialArmDyn(SerialArm):
    """
    SerialArmDyn class represents serial arms with dynamic properties and is used to calculate forces, torques, accelerations,
    joint forces, etc. using the Newton-Euler and Euler-Lagrange formulations. It inherits from the previously defined kinematic
    robot arm class "SerialArm". 
    """

    def __init__(self, 
                 dh, 
                 jt=None, 
                 base=eye, 
                 tip=eye, 
                 joint_limits=None,
                 mass=None,
                 r_com=None,
                 link_inertia=None,
                 motor_inertia=None,
                 joint_damping=None):

        SerialArm.__init__(self, dh, jt, base, tip, joint_limits)
        self.mass = mass
        self.r_com = r_com
        self.link_inertia = link_inertia
        self.motor_inertia = motor_inertia
        if joint_damping is None:
            self.B = np.zeros((self.n, self.n))
        else:
            self.B = np.diag(joint_damping)

    def rne(self, q, qd, qdd, 
            Wext=np.zeros((6,1)),
            g=np.zeros((3, 1)),
            omega_base=np.zeros((3, 1)),
            alpha_base=np.zeros((3, 1)),
            v_base=np.zeros((3, 1)),
            acc_base=np.zeros((3, 1))):

        """
        tau, W = RNE(q, qd, qdd):
        returns the torque in each joint (and the full wrench at each joint) given the joint configuration, velocity, and accelerations
        Args:
            q:
            qd:
            qdd:

        Returns:
            tau: torques or forces at joints (assuming revolute joints for now though)
            wrenches: force and torque at each joint, and for joint i, the wrench is in frame i


        We start with the velocity and acceleration of the base frame, v0 and a0, and the joint positions, joint velocities,
        and joint accelerations (q, qd, qdd).

        For each joint, we find the new angular velocity, w_i = w_(i-1) + z * qdot_(i-1)
        v_i = v_(i-1) + w_i x r_(i-1, com_i)


        if motor inertia is None, we don't consider it. Solve for now without motor inertia. The solution will provide code for motor inertia as well. 
        """

        omegas = []
        alphas = []
        v_ends = []
        v_coms = []
        acc_ends = []
        acc_coms = []

        ## Solve for needed angular velocities, angular accelerations, and linear accelerations
        ## If helpful, you can define a function to call here so that you can debug the output more easily. 
        for i in range(0, self.n):
            pass

        ## Now solve Kinetic equations by starting with forces at last link and going backwards
        ## If helpful, you can define a function to call here so that you can debug the output more easily. 
        Wrenches = [np.zeros((6,1))] * (self.n + 1)
        tau = [0] * self.n

        for i in range(self.n - 1, -1, -1):  # Index from n-1 to 0
            pass
            
        return tau, Wrenches



if __name__ == '__main__':

    ## this just gives an example of how to define a robot, this is a planar 3R robot.
    dh = [[0, 0, 1, 0],
          [0, 0, 1, 0],
          [0, 0, 1, 0]]

    joint_type = ['r', 'r', 'r']

    link_masses = [1, 1, 1]

    # defining three different centers of mass, one for each link
    r_coms = [np.array([-0.5, 0, 0]), np.array([-0.5, 0, 0]), np.array([-0.5, 0, 0])]

    link_inertias = []
    for i in range(len(joint_type)):
        iner = link_masses[i] / 12 * dh[i][2]**2

        # this inertia tensor is only defined as having Iyy, and Izz non-zero
        link_inertias.append(np.array([[0, 0, 0], [0, iner, 0], [0, 0, iner]]))


    arm = SerialArmDyn(dh,
                       jt=joint_type,
                       mass=link_masses,
                       r_com=r_coms,
                       link_inertia=link_inertias)

    # once implemented, you can call arm.RNE and it should work. 
    q = [np.pi/4.0]*3
    qd = [0.2]*3
    qdd = [0.05]*3
    arm.rne(q, qd, qdd)
