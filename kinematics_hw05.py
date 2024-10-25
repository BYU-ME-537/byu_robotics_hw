    # copy this code into the SerialArm __init__ function (found right after the "class SerialArm:" statement)
    # to define these variables required by the visualization for the arm
        self.reach = 0
        for i in range(self.n):
            self.reach += np.sqrt(self.dh[i][1]**2 + self.dh[i][2]**2)

        self.max_reach = 0.0
        for dh in self.dh:
            self.max_reach += norm(np.array([dh[1], dh[2]]))



    ## copy this function into the main SerialArm class and complete the TODO below
    def jacob(self, q, index=None, base=False, tip=False):
        """
        J = arm.jacob(q)
        Description:
        Returns the geometric jacobian for the frame defined by "index", which corresponds
        to a frame on the arm, with the arm in a given configuration defined by "q"

        Parameters:
        q - list or numpy array of joint positions
        index - integer, which joint frame at which to calculate the Jacobian

        Returns:
        J - numpy matrix 6xN, geometric jacobian of the robot arm
        """


        if index is None:
            index = self.n
        elif index > self.n:
            print("WARNING: Index greater than number of joints!")
            print(f"Index: {index}")

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
