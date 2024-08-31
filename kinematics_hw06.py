# insert this function into your SerialArm class and complete it. 
# Please keep the function definition, and what it returns the same. 
    def ik_position(self, target, q0=None, method='J_T', force=True, tol=1e-4, K=None, kd=0.001, max_iter=100):
        """
        (qf, ef, iter, reached_max_iter, status_msg) = arm.ik2(target, q0=None, method='jt', force=False, tol=1e-6, K=None)
        Description:
            Returns a solution to the inverse kinematics problem finding
            joint angles corresponding to the position (x y z coords) of target

        Args:
            target: 3x1 numpy array that defines the target location. 

            q0: length of initial joint coordinates, defaults to q=0 (which is
            often a singularity - other starting positions are recommended)

            method: String describing which IK algorithm to use. Options include:
                - 'pinv': damped pseudo-inverse solution, qdot = J_dag * e * dt, where
                J_dag = J.T * (J * J.T + kd**2)^-1
                - 'J_T': jacobian transpose method, qdot = J.T * K * e

            force: Boolean, if True will attempt to solve even if a naive reach check
            determines the target to be outside the reach of the arm

            tol: float, tolerance in the norm of the error in pose used as termination criteria for while loop

            K: 3x3 numpy matrix. For both pinv and J_T, K is the positive definite gain matrix used for both. 

            kd: is a scalar used in the pinv method to make sure the matrix is invertible. 

            max_iter: maximum attempts before giving up.

        Returns:
            qf: 6x1 numpy matrix of final joint values. If IK fails to converge the last set
            of joint angles is still returned

            ef: 3x1 numpy vector of the final error

            count: int, number of iterations

            flag: bool, "true" indicates successful IK solution and "false" unsuccessful

            status_msg: A string that may be useful to understanding why it failed. 
        """
        # Fill in q if none given, and convert to numpy array 
        if isinstance(q0, np.ndarray):
            q = q0
        elif q0 == None:
            q = np.array([0.0]*self.n)
        else:
            q = np.array(q0)

        # initializing some variables in case checks below don't work
        error = None
        count = 0

        # Try basic check for if the target is in the workspace.
        # Maximum length of the arm is sum(sqrt(d_i^2 + a_i^2)), distance to target is norm(A_t)
        maximum_reach = 0
        for i in range(self.n):  # Add max length of each link
            maximum_reach = maximum_reach + np.sqrt(self.dh[i][1] ** 2 + self.dh[i][2] ** 2)

        pt = target  # Find distance to target
        target_distance = np.sqrt(pt[0] ** 2 + pt[1] ** 2 + pt[2] ** 2)

        if target_distance > maximum_reach and not force:
            print("WARNING: Target outside of reachable workspace!")
            return q, error, count, False, "Failed: Out of workspace"
        else:
            if target_distance > maximum_reach:
                print("Target out of workspace, but finding closest solution anyway")
            else:
                print("Target passes naive reach test, distance is {:.1} and max reach is {:.1}".format(
                    float(target_distance), float(maximum_reach)))

        if not isinstance(K, np.ndarray):
            return q, error, count, False,  "No gain matrix 'K' provided"



        # you may want to define some functions here to help with operations that you will 
        # perform repeatedly in the while loop below. Alternatively, you can also just define 
        # them as class functions and use them as self.<function_name>.

        # for example:
        # def get_error(q):
        #     cur_position = 
        #     e = 
        #     return e

        while np.linalg.norm(error) > tol and count < max_iter:
        
        # In this while loop you will update q for each iteration, and update, then
        # your error to see if the problem has converged. You may want to print the error
        # or the "count" at each iteration to help you see the progress as you debug. 
        # You may even want to plot an arm initially for each iteration to make sure 
        # it's moving in the right direction towards the target. 



        # when "while" loop is done, return the relevant info. 

        return (q, error, count, count < max_iter, 'No errors noted')


