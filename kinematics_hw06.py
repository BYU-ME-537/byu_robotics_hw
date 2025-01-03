# insert this function into your SerialArm class and complete it.
# Please keep the function definition, and what it returns the same.
    def ik_position(self, target: NDArray, q0: list[float]|NDArray|None=None,
                    method: str='J_T', force: bool=True, tol: float=1e-4,
                    K: NDArray=None, kd: float=0.001, max_iter: int=100,
                    debug: bool=False, debug_step: bool=False
                    ) -> tuple[NDArray, NDArray, int, bool]:
        """
        qf, error_f, iters, converged = arm.ik_position(target, q0, 'J_T', K=np.eye(3))

        Computes the inverse kinematics solution (position only) for a given target
        position using a specified method by finding a set of joint angles that
        place the end effector at the target position without regard to orientation.

        :param NDArray target: 3x1 numpy array that defines the target location.
        :param list[float] | NDArray | None q0: list or array of initial joint positions,
            defaults to q0=0 (which is often a singularity - other starting positions
            are recommended).
        :param str method: select which IK algorithm to use. Options include:
            - 'pinv': damped pseudo-inverse solution, qdot = J_dag * e * dt, where
            J_dag = J.T * (J * J.T + kd**2)^-1
            - 'J_T': jacobian transpose method, qdot = J.T * K * e
        :param bool force: specify whether to attempt to solve even if a naive reach
            check shows the target is outside the reach of the arm.
        :param float tol: tolerance in the norm of the error in pose used as
            termination criteria for while loop.
        :param NDArray K: 3x3 numpy array. For both pinv and J_T, K is the positive
            definite gain matrix.
        :param float kd: used in the pinv method to make sure the matrix is invertible.
        :param int max_iter: maximum attempts before giving up.
        :param bool debug: specify whether to plot the intermediate steps of the algorithm.
        :param bool debug_step: specify whether to pause between each iteration when debugging.

        :return qf: 6x1 numpy array of final joint values. If IK fails to converge
            within the max iterations, the last set of joint angles is still returned.
        :return error_f: 3x1 numpy array of the final positional error.
        :return iters: int, number of iterations taken.
        :return converged: bool, specifies whether the IK solution converged within
            the max iterations.
        """
        ###############################################################################################
        # the following lines of code are data type and error checking. You don't need to understand
        # all of it, but it is helpful to keep.
        if isinstance(q0, np.ndarray):
            q = q0
        elif q0 == None:
            q = np.array([0.0]*self.n)
        elif isinstance(q0, list):
            q = np.array(q0)
        else:
            raise TypeError("Invlid type for initial joint positions 'q0'")

        # Try basic check for if the target is in the workspace.
        # Maximum length of the arm is sum(sqrt(d_i^2 + a_i^2)), distance to target is norm(A_t)
        target_distance = np.linalg.norm(target)
        target_in_reach = target_distance <= self.reach
        if not force:
            assert target_in_reach, "Target outside of reachable workspace!"
        if not target_in_reach:
            print("Target out of workspace, but finding closest solution anyway")

        assert isinstance(K, np.ndarray), "Gain matrix 'K' must be provided as a numpy array"
        ###############################################################################################
        ###############################################################################################

        # you may want to define some functions here to help with operations that you will
        # perform repeatedly in the while loop below. Alternatively, you can also just define
        # them as class functions and use them as self.<function_name>.

        # for example:
        # def get_error(q):
        #     cur_position =
        #     e =
        #     return e

        iters = 0
        while np.linalg.norm(error) > tol and iters < max_iter:

        # In this while loop you will update q for each iteration, and update, then
        # your error to see if the problem has converged. You may want to print the error
        # or the "count" at each iteration to help you see the progress as you debug.
        # You may even want to plot an arm initially for each iteration to make sure
        # it's moving in the right direction towards the target.



        # when "while" loop is done, return the relevant info.
        return q, error, iters, iters < max_iter
