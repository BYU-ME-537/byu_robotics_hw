import sympy as sp
import kinematics as kin

class SerialArmDyn(kin.SerialArm):
    """
    SerialArmDyn class represents serial arms with dynamic properties and is used to calculate forces, torques, accelerations,
    joint forces, etc. using the Euler-Lagrange formulations.
    """
    def __init__(self, dh,
                 jt=None,
                 base=sp.eye(4),
                 tip=sp.eye(4),
                 joint_limits=None,
                 mass=None,
                 r_com=None,
                 link_inertia=None,
                 motor_inertia=None,
                 joint_damping=None,
                 radians=True,
                 grav = sp.Matrix(3,1, [0, 0, 0])):


        kin.SerialArm.__init__(self, dh, jt, base, tip, radians)
        self.masses = mass
        self.coms = r_com
        self.Is = link_inertia
        self.Ims = motor_inertia

        self.q_sym = sp.symbols('q_1:' + str(self.n + 1), real=True)
        self.qd_sym = sp.symbols('qd_1:' + str(self.n + 1), real=True)
        self.qdd_sym = sp.symbols('qdd_1:' + str(self.n + 1), real=True)
        self.g_sym = sp.Matrix(3, 1, sp.symbols('gx, gy, gz'), real=True)

        self.grav = grav
        self.generate_EL()


    def generate_EL(self):
        """
        generate_EL():
        returns nothing, but calculates and stores functions for "M(q)", "C(q,qdot)", and "G(q)"
        Args:
            None
        Returns:
            None
        """

        #define helper variables
        jacob_com = []
        T_jts = []

        # these are symbolic versions of all joint angles, and their time derivatives
        q = self.q_sym
        qd = self.qd_sym

        # now get the Jacobians at the COM of each link
        for i in range(self.n):
            T_jts.append(self.fk(q, i+1))
            T_i_in_base = T_jts[i]
            r_com_in_base_frame = T_i_in_base[0:3,0:3] @ self.coms[i]
            jacob_com_i = self.jacob_shift(q, sp.eye(3), r_com_in_base_frame, index=i+1)

            # using "sp.simplify" in any of the code below makes for slower code generation,
            # but about 20 times faster execution of the function after using "lambdify"
            jacob_com.append(sp.simplify(jacob_com_i.evalf()))

        M_EL = sp.zeros(self.n, self.n)
        # TODO - use one for loop to implement symbolic code for M(q)

        C_EL = sp.zeros(self.n, self.n)
        # TODO - use three nested for loops to implement symbolic code for C(q, qd)

        P = sp.Matrix([0])
        # TODO - use one for loop to calculate P

        G_EL = sp.zeros(arm.n,1)
        # TODO - use one for loop to calculate symbolic code for G(q)

        self.M = sp.lambdify([q], M_EL, 'numpy')
        self.C = sp.lambdify([q, qd], C_EL, 'numpy')
        self.G = sp.lambdify([q], G_EL, 'numpy')


if __name__ == '__main__':

    r = sp.Matrix(3, 1, [-0.5, 0, 0])
    a = 1
    m = 1
    I = sp.Matrix([[0, 0, 0],
                    [0, a ** 2 * m / 12, 0],
                    [0, 0, a ** 2 * m / 12]])

    dh = [[0, 0, a, 0],
          [0, 0, a, 0]]

    joint_type = ['r', 'r']
    link_masses = [m, m]
    r_coms = [r, r]
    link_inertias = [I, I]
    arm = SerialArmDyn(dh,
                       jt=joint_type,
                       mass=link_masses,
                       r_com=r_coms,
                       link_inertia=link_inertias)

    print(arm.M([0, 0]))
