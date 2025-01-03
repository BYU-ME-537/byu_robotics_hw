def se3(R: NDArray=np.eye(3), p: NDArray=np.zeros(3)) -> NDArray:
    """
    T = se3(R, p)

    Creates a 4x4 homogeneous transformation matrix "T" from a 3x3 rotation matrix
    and a position vector.

    :param NDArray R: 3x3 numpy array representing orientation, defaults to identity.
    :param NDArray p: numpy array representing position, defaults to [0, 0, 0].
    :return T: 4x4 numpy array representing the homogeneous transform.
    """
    # TODO - fill out "T"
    T =

    return T

def inv(T: NDArray) -> NDArray:
    """
    T_inv = inv(T)

    Returns the inverse transform to T.

    :param NDArray T: 4x4 homogeneous transformation matrix
    :return T_inv: 4x4 numpy array that is the inverse to T so that T @ T_inv = I
    """

    #TODO - fill this out
    R =
    p =
    R_inv =
    p_inv =
    T_inv =

    return T_inv
