    ## copy this function into the main SerialArm class and complete the TODO below
    def jacob(self, q: list[float]|NDArray, index: int|None=None, base: bool=False,
              tip: bool=False) -> NDArray:
        """
        J = arm.jacob(q)

        Calculates the geometric jacobian for a specified frame of the arm in a given configuration

        :param list[float] | NDArray q: joint positions
        :param int | None index: joint frame at which to calculate the Jacobian
        :param bool base: specify whether to include the base transform in the Jacobian calculation
        :param bool tip: specify whether to include the tip transform in the Jacobian calculation
        :return J: 6xN numpy array, geometric jacobian of the robot arm
        """

        if index is None:
            index = self.n
        assert 0 <= index <= self.n, 'Invalid index value!'

        # TODO - start by declaring a zero matrix that is the correct size for the Jacobian
        J = np.zeros()

        # TODO - find the current position of the point of interest (usually origin of frame "n")
        # using your fk function this will likely require additional intermediate variables than
        # what is shown here.
        pe =


        # TODO - calculate all the necessary values using your "fk" function, and fill every column
        # of the jacobian using this "for" loop. Functions like "np.cross" may also be useful.
        for i in range(index):
            # check if joint is revolute
            if self.jt[i] == 'r':


            # if not assume joint is prismatic
            else:


        return J
