# GitHub Copilot Instructions for ME 537 Robotics Homework

This repository contains homework template files for BYU's ME 537 Robotics course, implementing core robotics algorithms from transforms through dynamics and control.

## Architecture Overview

The codebase follows a **layered architecture** where each module builds upon previous ones:

- **transforms.py**: Foundation layer - rotation matrices, homogeneous transforms, Euler angles, quaternions, axis/angle representations
- **kinematics.py**: Serial arm forward kinematics using DH parameters, Jacobians. Contains `SerialArm` class
- **dynamics.py**: `SerialArmDyn` class inherits from `SerialArm`, adds RNE (Recursive Newton-Euler) and Euler-Lagrange dynamics
- **visualization.py**: PyQt5/OpenGL 3D visualization (`VizScene` class) with real-time arm animation
- **utility.py**: Mathematical utilities (angle wrapping, skew matrices, interpolation, matrix cleaning)

## Critical DH Parameter Convention

⚠️ **Non-standard DH parameter order**: `[theta, d, a, alpha]` - **NOT the textbook order!** This is the order of operations.

```python
dh = [[0, 0, 0.2, np.pi/2.0],    # [theta, d, a, alpha] for link 1
      [0, 0, 0.2, -np.pi/2.0],   # link 2
      [0, 0, 0.1, np.pi/2.0]]    # link 3

joint_type = ['r', 'r', 'p']  # 'r' = revolute, 'p' = prismatic
```

This convention is **explicitly documented in comments** throughout `kinematics.py` and differs from the course textbook.

## Class Inheritance Pattern

`SerialArmDyn` extends `SerialArm` by adding mass properties and dynamics methods:

```python
arm = dyn.SerialArmDyn(dh, 
                       jt=joint_type,
                       mass=link_masses, 
                       r_com=r_coms,           # center of mass positions
                       link_inertia=link_inertias)
```

Methods inherited from `SerialArm`: `fk()`, `jacob()`, etc.
New methods in `SerialArmDyn`: `rne()`, `M()`, `C()`, `G()` (when implemented)

## Dual Implementation Pattern: Numerical vs Symbolic

- **Main modules** (`kinematics.py`, `dynamics.py`): NumPy-based numerical computation
- **hw10_EL_code/** subdirectory: SymPy-based symbolic versions for Euler-Lagrange derivation
- Convert symbolic to numerical: `sp.lambdify(vars, expr, modules='numpy')`

```python
# Symbolic approach (hw10_EL_code/)
import sympy as sp
r_coms = [sp.Matrix(3, 1, [-0.2, 0, 0])]  # SymPy Matrix

# Numerical approach (main files)
r_coms = [np.array([-0.1, 0, 0])]  # NumPy array
```

## Visualization Workflow

Always follow this pattern for 3D visualization:

```python
from visualization import VizScene

viz = VizScene()
viz.add_arm(arm, draw_frames=True)  # Add robot arm
viz.add_frame(T, label='frame1')     # Add coordinate frame
viz.update(qs=[q])                   # Update with joint config
viz.hold()                           # Block until window closed
# or viz.hold(1/60) for 60Hz refresh in animation loop
```

For animation loops, use `viz.hold(dt)` instead of `time.sleep()` to control refresh rate.

## Homework Structure and TODO Pattern

Files contain `TODO` comments marking sections students must complete:

```python
def A(q: float) -> NDArray:
    # TODO - complete code that defines the "A" or "T" homogenous matrix
    T = 
    return T
```

- **hw01-hw02**: Basic transforms and rotations
- **hw03**: Homogeneous transforms (`se3()`, `inv()`)
- **hw04-hw05**: Forward kinematics, Jacobians
- **hw09**: Simulation and control (RNE dynamics, numerical integration with `scipy.integrate.solve_ivp`)
- **hw10**: Symbolic Euler-Lagrange dynamics (uses `hw10_EL_code/` subdirectory)

## Test File Validation Pattern

Test files (`hwXX_test_*.py`, `hwXX_test_*.ipynb`) contain **expected outputs as LaTeX-formatted matrices in comments**:

```python
# $$ J = \left[\begin{matrix}0 & 0 & 0 & 0\\0.5 & 0 & 0.1 & -1.0\\ ...
```

Run these to verify implementations match known correct solutions.

## Key Utility Functions

- **Angle handling**: Always use `utility.wrap_angle()` and `utility.wrap_relative()` for angle arithmetic (handles ±π discontinuities)
- **Matrix cleaning**: Apply `utility.clean_rotation_matrix(R)` to remove numerical noise (values < 1e-12 → 0)
- **Skew operator**: `utility.skew(vec3)` creates skew-symmetric matrix from 3-vector

## State Vector Convention for Dynamics

State vector is `[q, qd]` (positions then velocities):

```python
def robot_eom(t, x):
    q = x[:arm.n]        # First n elements: joint positions
    qd = x[arm.n:]       # Last n elements: joint velocities
    # Calculate qdd using dynamics
    return np.concatenate([qd, qdd])
```

Use with `scipy.integrate.solve_ivp()` for numerical integration.

## Transform Composition

Chain transforms using `@` operator: `T_final = T1 @ T2 @ T3`

Homogeneous transform structure:
```python
T[:3, :3]  # Rotation matrix (3x3)
T[:3, 3]   # Position vector (3,)
```

## Common Debugging Patterns and Errors

### Array Shape Mismatches
**Problem**: Most common error - mixing row vectors, column vectors, and scalars
```python
# Common mistakes:
q = [0, 0, 0]           # List - works for most functions
q = np.array([0, 0, 0]) # 1D array (3,) - works for most functions
q_vec = np.array([[0], [0], [0]])  # Column vector (3,1) - needed for matrix multiply

# Fix for symbolic/numeric conversion (hw10):
q_vec = np.array(q).reshape(3, 1)  # Convert list to column vector
```

**Error messages**:
- `operands could not be broadcast together` → Check if mixing (3,) and (3,1) arrays
- `matmul: Input operand has more dimensions than allowed` → Wrong matrix dimensions
- Use `.squeeze()` to remove singleton dimensions, `.reshape()` to fix shapes

### DH Parameter Convention Errors
**Problem**: Using textbook DH order instead of code convention
```python
# WRONG (textbook): [d, theta, a, alpha]
# CORRECT (this code): [theta, d, a, alpha]

# Double-check when joint isn't moving as expected in visualization
```

### Index Validation Assertions
Functions include extensive error checking - read the assertion messages:
```python
assert len(q) == self.n, "q must be the same size as the number of links!"
assert 0 <= start_frame <= end_frame <= self.n, "Invalid index values!"
```

### Visualization Not Appearing
**Common causes**:
1. Missing `viz.hold()` or `viz.hold(dt)` - window closes immediately
2. Not calling `viz.update()` in animation loop
3. PyQt5 not installed correctly - run `pip install -r requirements.txt` again

### Numerical Integration Issues (hw09)
**State vector ordering**: Must be `[q, qd]` or `[qd, q]` consistently
```python
def robot_eom(t, x):
    q = x[:arm.n]   # First half: positions
    qd = x[arm.n:]  # Second half: velocities
    # Calculate qdd using dynamics
    return np.concatenate([qd, qdd])  # Must match input ordering!
```

**Common mistake**: Forgetting to specify `axis` parameter in `np.gradient()`:
```python
qd = np.gradient(q, t, axis=0)  # axis=0 for rows, axis=1 for columns
```

### Angle Wrapping Issues
**Problem**: Angles jumping between ±π causing discontinuities
```python
# WRONG: q_error = q_desired - q_actual  # Can jump from π to -π
# CORRECT:
from utility import wrap_relative, wrap_angle
q_error = wrap_relative(q_desired, q_actual) - q_actual
```

### Matrix Not Returning Expected Values
**Solution**: Use `utility.clean_rotation_matrix(R)` to zero out numerical noise:
```python
R = clean_rotation_matrix(R)  # Values < 1e-12 → 0, near 1 → 1
```

### Forward Kinematics Returns Identity
**Debugging checklist**:
1. Did you implement `dh2A()` function? (has `TODO` comment)
2. Did you append functions to `self.transforms` list in `__init__`?
3. Did you implement the `fk()` method loop over transforms?
4. Check with `print(arm.transforms)` - should have N functions, not empty list

### SymPy vs NumPy Confusion (hw10)
**Problem**: Mixing symbolic and numerical computation
```python
# Symbolic (hw10_EL_code/)
import sympy as sp
r_com = sp.Matrix(3, 1, [-0.2, 0, 0])  # SymPy Matrix
# Convert to callable function:
func = sp.lambdify([q1, q2], expression, modules='numpy')

# Numerical (main files)
r_com = np.array([-0.2, 0, 0])  # NumPy array
```

## Development Workflow

1. **Environment setup**: `pip install -r requirements.txt` in VS Code with Python extension
2. **Test setup**: Run `hw01.py` - should display animated coordinate frames in PyQt5 window
3. **Implementation pattern**: Complete `TODO` sections in order (transforms → kinematics → dynamics)
4. **Validation**: Run corresponding `hwXX_test_*.py` files to check against expected outputs
5. **Debugging**: If stuck, check shape of arrays with `.shape`, print intermediate values, verify DH convention
