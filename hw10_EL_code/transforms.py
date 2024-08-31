import sympy as sp
import mpmath as mp

from _Transforms import SO2 as SO2Base
from _Transforms import SO3 as SO3Base
from _Transforms import SE3 as SE3Base


class SO2(SO2Base):
    """
    A class representing a 2D rotation, or the SO2 Lie group

    Attributes:
    angle - the angle of rotation, can be thought of as a rotation about the z axis
    R - the 2x2 sympy matrix representing the rotation

    Methods:
    init - creates an instance of the class
    inv - returns the inverse of the rotation

    Overloaded Operators:
    @ - composes or multiplies rotation matrices
    """
    def __init__(self, angle, radians=False):
        """ 
        R = SO2(angle, radians=False)
        Parameters:
        angle - float or sympy symbol, the angle about the z axis to rotate
        radians - bool, True if angle is in radians and false if in degrees, defaults to False

        Returns:
        instance of SO2
         """
        SO2Base.__init__(self)  # This line imports some functionality from the base class, don't worry about it!

        # convert to radians
        if not radians:
            angle = mp.radians(angle)

        self.angle = angle
        
        # construct the rotation matrix
        self.R = sp.zeros(2)
        self.R[0,0] = sp.cos(angle)
        self.R[0,1] = -sp.sin(angle)
        self.R[1,0] = sp.sin(angle)
        self.R[1,1] = sp.cos(angle)

    def inv(self):
        """
        Rinv = R.inv()
        Returns a new instance of SO2 which is the inverse of the original instance
        """
        # if you tried to define this using something like self.R.T appropriately, that could also work. 
        return SO2(-self.angle, radians=True)


class SO3(SO3Base):
    """
    A class representing a general 3D rotation or orientation, or an SO3 Lie group object

    Attributes:
    R - The 3x3 sympy matrix representing the rotation

    Methods:
    init - creates instance of class
    inv - returns an inverse of the rotation matrix
    rpy - returns an rpy representation of the rotation matrix
    axis - returns an axis angle representation of the rotation matrix
    quaternion - returns a quaternion representation of the rotation matrix

    Overloaded Operators:
    @ - composes multiple rotations
    """
    def __init__(self, R):
        """
        Rot = SO3(R)
        Description:
        Constructor for the SO3 class, representing rotations in 3D

        Parameters:
        R - 3x3 sympy matrix, should be orthonormal

        Returns:
        Rot - SO3 instance
        """

        self.R = R

    def inv(self):
        """
        R' = R.inv()
        Description:
        returns inverse of rotation matrix

        Parameters:
        None

        Returns:
        R' - SO3 Object which is the inverse of R, so that R @ R' = I
        """

        # Take transpose of rotation matrix for inverse
        R = self.R.T
        return SO3(R)

    def rpy(self):
        """
        rpy = R.rpy()
        Description:
        Returns the roll-pitch-yaw representation of the SO3 object

        Parameters:
        None

        Returns:
        rpy - 1 x 3 Sympy Matrix, containing <roll pitch yaw> coordinates (in radians)
        """
        # follow formula in book, see equation (2.22) on pg. 52 using functions like "sp.atan2" 
        # for the arctangent and "**2" for squared terms. 
        rpy = sp.Matrix([sp.atan2(self.R[1,0], self.R[0,0]), 
                    sp.atan2(-self.R[2,0], sp.sqrt(self.R[2,1]**2+self.R[2,2]**2)),
                    sp.atan2(self.R[2,1], self.R[2,2])])

        return rpy
    
    def axis(self):
        """
        axis_angle = R.axis()
        Description:
        Returns an axis angle representation of pose

        Parameters:
        None

        Returns:
        axis_angle - 1 x 4 sympy matrix, containing  the axis angle representation
        in the form: <angle, rx, ry, rz>
        """
        # see equation (2.27) and (2.28) on pg. 54
        ang = sp.acos(0.5*(self.R[0,0]+self.R[1,1]+self.R[2,2]-1))
        axis_angle = sp.Matrix([ang,
                        1/(2*sp.sin(ang))*(self.R[2,1]-self.R[1,2]),
                        1/(2*sp.sin(ang))*(self.R[0,2]-self.R[2,0]),
                        1/(2*sp.sin(ang))*(self.R[1,0]-self.R[0,1])])
        return axis_angle

    def quaternion(self):
        """
        quaternion = R.quaternion()
        Description:
        Returns a quaternion representation of pose

        Parameters:
        None

        Returns:
        quaternion - 1 x 4 sympy matrix, quaternion representation of pose in the 
        format <nu, ex, ey, ez>
        """

        # see equation (2.34) and (2.35) on pg. 55
        quaternion = sp.Matrix([0.5*sp.sqrt(self.R[0,0]+self.R[1,1]+self.R[2,2]+1),
                             0.5*sp.sign(self.R[2,1]-self.R[1,2])*sp.sqrt(self.R[0,0]-self.R[1,1]-self.R[2,2]+1),
                             0.5*sp.sign(self.R[0,2]-self.R[2,0])*sp.sqrt(self.R[1,1]-self.R[2,2]-self.R[0,0]+1),
                             0.5*sp.sign(self.R[2,1]-self.R[1,2])*sp.sqrt(self.R[2,2]-self.R[0,0]-self.R[1,1]+1)])
        return quaternion

    def euler(self, order, radians=False):
        """
        e = R.euler(order)
        Description:
            euler returns euler angles in whatever order is specified by the string in order

        Parameters:
            order - string, must be in the form 'axis1axis2axis3', e.g. 'xyz', 'xyx', 'zyz', etc, and cannot contain
            two consecutive rotations about the same axis
            radians - bool, radians or degrees

        Returns:
            e - 1 x 3 sympy matrix of the euler angles specified

        Notes:
            Lets denote the axis order as 'ABC' and add a ' symbol to each axis which represents how many rotations that
            axis has undergone. Thus, we start with an axis ABC (noting that A and C may be the same, e.g. this may not
            be a full description of the x y and z axes), and we will end with an axis A''B''C''.

            Working from the last rotation forwards, we will end with a rotation about C'' that aligns the other two
            axes with the final transform axes. Let's assume that we will find this rotation by looking at how the
            B' axis needs to rotate to line up with the B'' axis (again, there is only one rotation to make here so
            even though we are only looking at B' to B'', this rotation will also align the third axis). So our last
            rotation will be about C'', aligning B' with B''. In order for a rotation about C'' to be able to do this,
            C'' must already be perpendicular to B''. So our second rotation must align C' with C''. In order to do this
            B' must already by normal to C' and C'', so our first rotation must be about the A axis to align B' normal
            to C' and C''

            We first rotate about A and align B' with A x C''*, **
            Then we rotate about B' to align C' with C''**,
            Then we rotate about C'' to align B' with B''

            * The direction of A x C'' depends which axes, A and C, we have chosen. Ordering the unit axes from least to
            greatest as x < y < z, we will say that if A < C, we want to align B' with A x C'', else align B' with
            -A x C''

            ** We need to cover the case where A x C'' = 0, which corresponds to a rotation only about A. In this case
            our sequence is simple: We rotate about A to align B with B''
        """

        # Parse the axis order input, and assign 0 to x, 1 to y, and 2 to z, so that we can index the i vector with
        # R[axis1, :]

        if order[0] == 'x':
            rotA = rotx
            axis1 = 0
        elif order[0] == 'y':
            rotA = roty
            axis1 = 1
        else:
            rotA = rotz
            axis1 = 2

        if order[1] == 'x':
            rotB = rotx
            axis2 = 0
        elif order[1] == 'y':
            rotB = roty
            axis2 = 1
        else:
            rotB = rotz
            axis2 = 2

        if order[2] == 'x':
            rotC = rotx
            axis3 = 0
        elif order[2] == 'y':
            rotC = roty
            axis3 = 1
        else:
            rotC = rotz
            axis3 = 2

        # If we are ordering axes in reverse order, we need to take that into account and flip the sign of the
        # cross product, see doc string for more detail
        if axis1 >= axis3:
            s = -1
        else:
            s = 1

        # R is initially [A, B, C] and Rf is [A'', B'', C'']. Each rotation modified R; our sequence looks like this:
        # R = [A, B, C]. Rotate about R[A]
        # R = [A, B', C']. Rotate about R[B']
        # R = [A', B', C'']. Rotate about R[C'']
        # R = [A'', B'', C''] = Rf
        R = SO3(sp.eye(3))
        Rf = self

        v = Rf[:,axis3].cross(s * R[:,axis1])
        if v.norm() < 0.001:  # This indicates a rotation about the A axis ONLY.
            th1 = sp.acos(R[:,axis2].dot(Rf[:, axis2]))
            th2 = 0
            th3 = 0
            R = R @ rotA(th1)
        else:
            v = v / v.norm()
            th1 = sp.acos(R[:,axis2].dot(v))
            R = R @ rotA(th1)

            th2 = sp.acos(R[:,axis3].dot(Rf[:,axis3]))
            R = R @ rotB(th2)

            th3 = sp.acos(R[:, axis2].dot(Rf[:, axis2]))
            R = R @ rotC(th3)

        if not radians:
            th1 = mp.degrees(th1)
            th2 = mp.degrees(th2)
            th3 = mp.degrees(th3)

        return sp.Matrix(1, 3, [th1, th2, th3])




def rotx(theta, radians=False):
    """
    R = rotx(theta, radians=False)
    Description:
    Returns an SO3 Object representing a rotation about the x axis

    Parameters:
    theta - float or sympy symbol represention rotation about the x axis
    radians - bool, indicates if theta is in degrees or radians

    Returns:
    T - SO3 object
    """
    if not radians:
        theta = mp.radians(theta)
    R = sp.Matrix([[1, 0, 0],[0, sp.cos(theta), -sp.sin(theta)], [0, sp.sin(theta), sp.cos(theta)]])
    return SO3(R)

def roty(theta, radians=False):
    """
    R = rotx(theta, radians=False)
    Description:
    Returns an SO3 Object representing a rotation about the y axis

    Parameters:
    theta - float or sympy symbol represention rotation about the y axis
    radians - bool, indicates if theta is in degrees or radians

    Returns:
    T - SO3 object
    """
    if not radians:
        theta = mp.radians(theta)
    R = sp.Matrix([[sp.cos(theta), 0, sp.sin(theta)], [0, 1, 0], [-sp.sin(theta), 0, sp.cos(theta)]])
    return SO3(R)

def rotz(theta, radians=False):
    """
    R = rotx(theta, radians=False)
    Description:
    Returns an SO3 Object representing a rotation about the z axis

    Parameters:
    theta - float or sympy symbol represention rotation about the z axis
    radians - bool, indicates if theta is in degrees or radians

    Returns:
    T - SO3 object
    """
    if not radians:
        theta = mp.radians(theta)
    R = sp.Matrix([[sp.cos(theta), -sp.sin(theta), 0], [sp.sin(theta), sp.cos(theta), 0], [0, 0, 1]])
    return SO3(R)

def euler2R(th1, th2, th3, order='xyx', radians=True):
    """
    R = euler2R(th1, th2, th3, order='xyx', radians=True)
    Description:
    Returns an SO3 object of the rotation specified by the euler angles, we assume in all cases
    that these are defined about the "current axis," which is why there are only 12 versions 
    (instead of the 24 possiblities noted in the course slides). 

    Parameters:
    th1, th2, th3 - float or sympy symbol, angles of rotation
    order - string, specifies the euler rotation to use, for example 'xyx', 'zyz', etc.
    radians - bool, if true th1-th3 are assumed to be in radians, if false they are in degrees
    
    Returns:
    R - SO3 Object
    """
    if not radians:
        th1 = mp.radians(th1)
        th2 = mp.radians(th2)
        th3 = mp.radians(th3)

    if order == 'xyx':
        R = rotx(th1, radians=radians) @ roty(th2, radians=radians) @ rotx(th3, radians=radians)
    elif order == 'xyz':
        R = rotx(th1, radians=radians) @ roty(th2, radians=radians) @ rotz(th3, radians=radians)
    elif order == 'xzx':
        R = rotx(th1, radians=radians) @ rotz(th2, radians=radians) @ rotx(th3, radians=radians)
    elif order == 'xzy':
        R = rotx(th1, radians=radians) @ rotz(th2, radians=radians) @ roty(th3, radians=radians)
    elif order == 'yxy':
        R = roty(th1, radians=radians) @ rotx(th2, radians=radians) @ roty(th3, radians=radians)
    elif order == 'yxz':
        R = roty(th1, radians=radians) @ rotx(th2, radians=radians) @ rotz(th3, radians=radians)
    elif order == 'yzx':
        R = roty(th1, radians=radians) @ rotz(th2, radians=radians) @ rotx(th3, radians=radians)
    elif order == 'yzy':
        R = roty(th1, radians=radians) @ rotz(th2, radians=radians) @ roty(th3, radians=radians)
    elif order == 'zxy':
        R = rotz(th1, radians=radians) @ rotx(th2, radians=radians) @ roty(th3, radians=radians)
    elif order == 'zxz':
        R = rotz(th1, radians=radians) @ rotx(th2, radians=radians) @ rotz(th3, radians=radians)
    elif order == 'zyx':
        R = rotz(th1, radians=radians) @ roty(th2, radians=radians) @ rotx(th3, radians=radians)
    elif order == 'zyz':
        R = rotz(th1, radians=radians) @ roty(th2, radians=radians) @ rotz(th3, radians=radians)

    return R
    
def axis2R(angle, rx, ry, rz, radians=True):
    """
    R = axis2R(angle, rx, ry, rz, radians=True)
    Description:
    Returns an SO3 object of the rotation specified by the axis-angle

    Parameters:
    angle - float or sympy symbol, the angle to rotate about the axis
    rx, ry, rz - components of the unit axis to rotate about
    radians - bool, if true angle is assumed to be in radians, if false it is in degrees
    
    Returns:
    R - SO3 Object
    """
    if not radians:
        angle = mp.radians(angle)

    # follow formula in book, see equation (2.25) on pg. 54
    c = sp.cos(angle)
    s = sp.sin(angle)
    R = sp.Matrix([[rx**2 * (1-c) + c, rx*ry*(1-c)-rz*s, rx*rz*(1-c)+ry*s],
                        [rx*ry*(1-c)+rz*s, ry**2 * (1-c) + c, ry*rz*(1-c)-rx*s],
                        [rx*rz*(1-c)-ry*s, ry*rz*(1-c)+rx*s, rz**2 * (1-c) + c]])
    return SO3(R)

def quaternion2R(nu, ex, ey, ez):
    """
    R = quaternion2R(nu, ex, ey, ez, radians=True)
    Description:
    Returns an SO3 object of the rotation specified by the quaternion

    Parameters:
    nu, ex, ey, ez - floats or sympy symbols defining the quaternion
    
    Returns:
    R - SO3 Object
    """
    R = sp.Matrix([[2*(nu**2+ex**2)-1, 2*(ex*ey-nu*ez), 2*(ex*ez+nu*ey)],
                    [2*(ex*ey+nu*ez), 2*(nu**2+ey**2)-1, 2*(ey*ez-nu*ex)],
                    [2*(ex*ez-nu*ey), 2*(ey*ez+nu*ex), 2*(nu**2+ez**2)-1]])
    return SO3(R)


class SE3(SE3Base):
    """
    SE3 - A class representing pose in 3D space, or the SE3 Lie group
    Attributes:
    R - A 3x3 sympy matrix representing the orientation in 3D space of the frame
    p - A 3x1 sympy matrix representing the position in 3D space of the frame
    A - A 4x4 sympy matrix representing the pose in 3D space

    Methods:
    init - returns an instance of the class
    inv - returns the inverse of the transform

    Overloaded Operators:
    '@' - used to compose transformations
    """
    def __init__(self, R=sp.eye(3), p=sp.zeros(3,1)):
        """
        T = SE3(R=sp.eye(3), p=sp.zeros(3,1))
        Description:
        Constructor for the SE3 class

        Parameters:
        R - 3x3 sympy matrix representing orientation, defaults to identity
        p = 3x1 sympy matrix representing position, defaults to <0, 0, 0>^T

        Returns:
        T - SE3 instance
        """
        SE3Base.__init__(self)  # This just calls some stuff from the base class, don't worry about it

        if isinstance(R, SO3):  # If R is an SO3 object, pull the sympy matrix out
            R = R.R

        self.R = R
        self.p = p
        self.A = sp.eye(4)
        self.A[0:3, 0:3] = R
        self.A[0:3, 3] = p

    def inv(self):
        """
        Tinv = T.inv()
        Description:
        Returns the inverse transform to T

        Parameters:
        None

        Returns:
        Tinv - SE3 object, inverse to T so that T @ Tinv = I
        """
        A = self.A
        R = A[0:3, 0:3].T
        p = -R @ A[0:3, 3]
        return SE3(R, p)