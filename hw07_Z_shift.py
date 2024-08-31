from utility import skew

    def Z_shift(self, R=np.eye(3), p=np.zeros(3,), p_frame='i'):

        """
        Z = Z_shift(R, p, p_frame_order)
        Description: 
            Generates a shifting operator (rotates and translates) to move twists and Jacobians 
            from one point to a new point defined by the relative transform R and the translation p. 

        Parameters:
            R - 3x3 numpy array, expresses frame "i" in frame "j" (e.g. R^j_i)
            p - 3x1 numpy array length 3 iterable, the translation from the initial Jacobian point to the final point, expressed in the frame as described by the next variable.
            p_frame - is either 'i', or 'j'. Allows us to define if "p" is expressed in frame "i" or "j", and where the skew symmetrics matrix should show up. 

        Returns:
            Z - 6x6 numpy array, can be used to shift a Jacobian, or a twist
        """

        # generate our skew matrix
        S = skew(p)

        if p_frame == 'i':
            Z = 
        elif p_frame == 'j':
            Z = 
        else:
            Z = None

        return Z
