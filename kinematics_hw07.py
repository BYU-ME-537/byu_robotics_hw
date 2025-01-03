# Copy this import into kinematics.py:
from utility import skew

    ## copy this function into the main SerialArm class and complete the TODO below
    def Z_shift(self, R: NDArray=np.eye(3), p: NDArray=np.zeros(3,), p_frame: str='i'):
        """
        Z = Z_shift(R, p, p_frame)

        Generates a shifting operator (rotates and translates) to move twists and
        Jacobians from one point to a new point defined by the relative transform
        R and the translation p.

        :param NDArray R: 3x3 array that expresses frame "i" in frame "j" (e.g. R^j_i).
        :param NDArray p: (3,) array (or iterable), the translation from the initial
            Jacobian point to the final point, expressed in the frame as described
            by the next variable.
        :param str p_frame: is either 'i', or 'j'. Allows us to define if "p" is
            expressed in frame "i" or "j", and where the skew symmetrics matrix
            should show up.
        :return Z: 6x6 numpy array, can be used to shift a Jacobian, or a twist.
        """

        # generate our skew matrix
        S = skew(p)

        if p_frame == 'i':
            Z =
        elif p_frame == 'j':
            Z =
        else:
            raise ValueError("p_frame must be either 'i' or 'j'")

        return Z
