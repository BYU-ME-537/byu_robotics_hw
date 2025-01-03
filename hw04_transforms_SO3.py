"""
SO(3) conversion code to convert between different SO(3) representations.

Copy this file into your 'transforms.py' file at the bottom.
"""


def R2rpy(R: NDArray) -> NDArray:
    """
    rpy = R2rpy(R)

    Returns the roll-pitch-yaw representation of the SO3 rotation matrix.

    :param NDArray R: 3x3 Numpy array for any rotation.
    :return rpy: Numpy array, containing [roll, pitch, yaw] coordinates (in radians).
    """

    # follow formula in book, use functions like "np.atan2"
    # for the arctangent and "**2" for squared terms.
    # TODO - fill out this equation for rpy

    roll =
    pitch =
    yaw =

    return np.array([roll, pitch, yaw])


def R2axis(R: NDArray) -> NDArray:
    """
    axis_angle = R2axis(R)

    Returns an axis angle representation of a SO(3) rotation matrix.

    :param NDArray R: 3x3 rotation matrix.
    :return axis_angle: numpy array containing the axis angle representation
        in the form: [angle, rx, ry, rz]
    """

    # see equation (2.27) and (2.28) on pg. 54, using functions like "np.acos," "np.sin," etc.
    ang = # TODO - fill out here.
    axis_angle = np.array([ang,
                            , # TODO - fill out here, each row will be a function of "ang"
                            ,
                            ])

    return axis_angle


def axis2R(angle: float, axis: NDArray) -> NDArray:
    """
    R = axis2R(angle, axis)

    Returns an SO3 object of the rotation specified by the axis-angle.

    :param float angle: the angle to rotate about the axis (in radians).
    :param NDArray axis: components of the unit axis about which to rotate as
        a numpy array [rx, ry, rz].
    :return R: 3x3 numpy array representing the rotation matrix.
    """
    # TODO fill this out
    R =
    return clean_rotation_matrix(R)


def R2quat(R: NDArray) -> NDArray:
    """
    quaternion = R2quat(R)

    Returns a quaternion representation of pose.

    :param NDArray R: 3x3 rotation matrix.
    :return quaternion: numpy array for the quaternion representation of pose in
        the format [nu, ex, ey, ez]
    """
    # TODO, see equation (2.34) and (2.35) on pg. 55, using functions like "sp.sqrt," and "sp.sign"

    return np.array([,
                     ,
                     ,
                     ])


def quat2R(q: NDArray) -> NDArray:
    """
    R = quat2R(q)

    Returns a 3x3 rotation matrix from a quaternion.

    :param NDArray q: [nu, ex, ey, ez ] - defining the quaternion.
    :return R: numpy array, 3x3 rotation matrix.
    """
    # TODO, extract the entries of q below, and then calculate R
    nu =
    ex =
    ey =
    ez =
    R =
    return clean_rotation_matrix(R)


def euler2R(th1: float, th2: float, th3: float, order: str='xyz') -> NDArray:
    """
    R = euler2R(th1, th2, th3, order='xyz')

    Returns a 3x3 rotation matrix as specified by the euler angles, we assume in all cases
    that these are defined about the "current axis," which is why there are only 12 versions
    (instead of the 24 possiblities noted in the course slides).

    :param float th1: angle of rotation about 1st axis (rad)
    :param float th2: angle of rotation about 2nd axis (rad)
    :param float th3: angle of rotation about 3rd axis (rad)
    :param str order: specifies the euler rotation to use, for example 'xyx', 'zyz', etc.
    :return R: 3x3 numpy array, the rotation matrix.
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
        raise ValueError("Invalid Order!")

    return clean_rotation_matrix(R)
