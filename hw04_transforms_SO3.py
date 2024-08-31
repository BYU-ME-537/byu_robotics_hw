"""
SO(3) conversion code to convert between different SO(3) representations. 

Copy this file into your 'transforms.py' file at the bottom. 
"""

import numpy as np
from numpy import sin, cos, sqrt

def R2rpy(R):
    """
    rpy = R2rpy(R)
    Description:
    Returns the roll-pitch-yaw representation of the SO3 rotation matrix

    Parameters:
    R - 3 x 3 Numpy array for any rotation

    Returns:
    rpy - 1 x 3 Numpy Matrix, containing <roll pitch yaw> coordinates (in radians)
    """
    
    # follow formula in book, use functions like "np.atan2" 
    # for the arctangent and "**2" for squared terms. 
    # TODO - fill out this equation for rpy

    roll = 
    pitch = 
    yaw = 

    return np.array([roll, pitch, yaw])


def R2axis(R):
    """
    axis_angle = R2axis(R)
    Description:
    Returns an axis angle representation of a SO(3) rotation matrix

    Parameters:
    R

    Returns:
    axis_angle - 1 x 4 numpy matrix, containing  the axis angle representation
    in the form: <angle, rx, ry, rz>
    """

    # see equation (2.27) and (2.28) on pg. 54, using functions like "np.acos," "np.sin," etc. 
    ang = # TODO - fill out here. 
    axis_angle = np.array([ang,
                            , # TODO - fill out here, each row will be a function of "ang"
                            ,
                            ])

    return axis_angle

def axis2R(ang, v):
    """
    R = axis2R(angle, rx, ry, rz, radians=True)
    Description:
    Returns an SO3 object of the rotation specified by the axis-angle

    Parameters:
    angle - float, the angle to rotate about the axis in radians
    v = [rx, ry, rz] - components of the unit axis about which to rotate as 3x1 numpy array
    
    Returns:
    R - 3x3 numpy array
    """
    # TODO fill this out 
    R = 
    return R

def R2quatuat(R):
    """
    quaternion = R2quat(R)
    Description:
    Returns a quaternion representation of pose

    Parameters:
    R

    Returns:
    quaternion - 1 x 4 numpy matrix, quaternion representation of pose in the 
    format [nu, ex, ey, ez]
    """
    # TODO, see equation (2.34) and (2.35) on pg. 55, using functions like "sp.sqrt," and "sp.sign"

    return np.array([,
                     ,
                     ,
                     ])
                    
def quat2R(q):
    """
    R = quat2R(q)
    Description:
    Returns a 3x3 rotation matrix

    Parameters:
    q - 4x1 numpy array, [nu, ex, ey, ez ] - defining the quaternion
    
    Returns:
    R - a 3x3 numpy array 
    """
    # TODO, extract the entries of q below, and then calculate R
    nu = 
    ex = 
    ey = 
    ez = 
    R =  
    return R


def euler2R(th1, th2, th3, order='xyz'):
    """
    R = euler2R(th1, th2, th3, order='xyz')
    Description:
    Returns a 3x3 rotation matrix as specified by the euler angles, we assume in all cases
    that these are defined about the "current axis," which is why there are only 12 versions 
    (instead of the 24 possiblities noted in the course slides). 

    Parameters:
    th1, th2, th3 - float, angles of rotation
    order - string, specifies the euler rotation to use, for example 'xyx', 'zyz', etc.
    
    Returns:
    R - 3x3 numpy matrix
    """

    # TODO - fill out each expression for R based on the condition 
    # (hint: use your rotx, roty, rotz functions)
    if order == 'xyx':
        R = 
    elif order == 'xyz':
        R = 
    elif order == 'xzx':
        R = 
    elif order == 'xzy':
        R = 
    elif order == 'yxy':
        R = 
    elif order == 'yxz':
        R = 
    elif order == 'yzx':
        R = 
    elif order == 'yzy':
        R = 
    elif order == 'zxy':
        R = 
    elif order == 'zxz':
        R = 
    elif order == 'zyx':
        R = 
    elif order == 'zyz':
        R = 
    else:
        print("Invalid Order!")
        return

    return R
