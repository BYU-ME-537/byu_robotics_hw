"""
Transforms Module - Contains code for:
- 2D Rotation Matrices
- 3D Rotation Matrices
- Homogeneous Transforms

John Morrell, August 21 2021
Tarnarmour@gmail.com
"""

import sympy as sp
import numpy as np
import mpmath as mp

class SO2:

    """
    A class representing a 2D rotation, or the SO2 Lie group

    Attributes:
    angle - the angle of rotation, can be thought of as a rotation about the z axis
    R - the 2x2 sympy matrix representing the rotation

    Methods:
    init - creates an instance of the class
    inv - returns the inverse of the rotation

    Overloaded Operators:
    @ - composes or multiplies rotation matrices
    """

    def __init__(self):
        """ 
        R = SO2(angle, radians=False)
        Parameters:
        angle - float or sympy symbol, the angle about the z axis to rotate
        radians - bool, True if angle is in radians and false if in degrees, defaults to False

        Returns:
        instance of SO2
         """

        self.angle = None
        self.R = None

    
    def __str__(self):
        """
        Provides definition for str() function, so that print() works
        """
        return "Rotation Matrix\n" + sp.pretty(sp.N(self.R, 4))

    def __matmul__(R1, R2):
        """
        Overloads the '@' operator, only defined for composition of two SO2 instances
        """
        if isinstance(R2, SO2):
            angle = R1.angle + R2.angle
            return R1.__class__(angle, radians=True)
        else:
            print("ERROR! Invalid operands for rotation composition!")
            return None

class SO3:
    """
    A class representing a general 3D rotation or orientation, or an SO3 Lie group object

    Attributes:
    R - The 3x3 sympy matrix representing the rotation
    self.rpy - a 3 element list representing the rotation as Roll Pitch Yaw angles, in degrees
    self.axis_angle - a 4 element list representing the rotation as an axis angle pair, [angle, rx, ry, rz]
    self.quaternion - a 4 element list representing the rotation as a quaternion

    Methods:
    init - creates instance of class
    inv - returns an inverse of the rotation matrix

    Overloaded Operators:
    @ - composes multiple rotations
    """

    def __init__(self):
        pass
        
    def __matmul__(R1, R2):
        """
        Overload the '@' operator, only valid for composition of two SO3 objects
        """
        if not isinstance(R2, SO3):
            print("Error! Invalid operand for rotation composition!")
            return None
        R = R1.R @ R2.R
        return R1.__class__(R)

    def __str__(self):
        """
        Provide definition for str() function, so that print() works
        """
        a = "Rotation Matrix:\n" + sp.pretty(sp.N(self.R, 4))
        return a

    def __getitem__(self, key):
        return self.R[key]

    def __setitem__(self, key, value):
        self.R[key] = value

    
class SE3:
    """
    SE3 - A class representing pose in 3D space, or the SE3 Lie group
    Attributes:
    R - A SO3 object representing the orientation of the frame
    p - A 3x1 sympy matrix representing the position in 3D space of the frame
    A - A 4x4 sympy matrix representing the pose in 3D space

    Methods:
    init - returns an instance of the class
    inv - returns the inverse of the transform

    Overloaded Operators:
    '@' - used to compose transformations
    """
    def __init__(self):
        pass
    
    def __matmul__(A1, A2):
        """
        Overloads the '@' operator, used to compose HomoegeneousTransfrom objects
        """
        if not isinstance(A2, SE3):
            print("Error! Invalid operand for transform composition!")
            return None
        
        A = A1.A @ A2.A
        R = A[0:3, 0:3]
        p = A[0:3, 3]
        return A1.__class__(R, p)
    
    def __str__(self):
        """
        Defines the str() function to allow print() to work
        """
        return "Homogeneous Transform:\n" + sp.pretty(sp.N(self.A,4))

    def __getitem__(self, key):
        return self.A[key]

    def __setitem__(self, key, value):
        A = self.A
        A[key] = value
        self.R = A[0:3, 0:3]
        self.p = A[0:3, 3]
        self.A = A


