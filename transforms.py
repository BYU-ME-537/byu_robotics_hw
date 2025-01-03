"""
Transforms Module - Contains code for to learn about rotations
and eventually homogenous transforms.

Empty outline derived from code written by John Morrell, former TA.
"""

import numpy as np
from numpy import sin, cos, sqrt
from numpy.typing import NDArray
from utility import clean_rotation_matrix


## 2D Rotations
def rot2(theta: float) -> NDArray:
    """
    R = rot2(th)

    :param float theta: angle of rotation (rad)
    :return R: 2x2 numpy array representing rotation in 2D by theta
    """

    ## TODO - Fill this out
    R =
    return clean_rotation_matrix(R)


## 3D Transformations
def rotx(theta: float) -> NDArray:
    """
    R = rotx(theta)

    :param float theta: angle of rotation (rad)
    :return R: 3x3 numpy array representing rotation about x-axis by amount theta
    """
    ## TODO - Fill this out
    R =

    return clean_rotation_matrix(R)


def roty(theta: float) -> NDArray:
    """
    R = roty(theta)

    :param float theta: angle of rotation (rad)
    :return R: 3x3 numpy array representing rotation about y-axis by amount theta
    """
    ## TODO - Fill this out
    R =

    return clean_rotation_matrix(R)


def rotz(theta: float) -> NDArray:
    """
    R = rotz(theta)

    :param float theta: angle of rotation (rad)
    :return R: 3x3 numpy array representing rotation about z-axis by amount theta
    """
    ## TODO - Fill this out
    R =

    return clean_rotation_matrix(R)


# inverse of rotation matrix
def rot_inv(R: NDArray) -> NDArray:
    '''
    R_inv = rot_inv(R)

    :param NDArray R: 2x2 or 3x3 numpy array representing a proper rotation matrix
    :return R_inv: 2x2 or 3x3 inverse of the input rotation matrix
    '''
    ## TODO - Fill this out
    return
