import jax
import jax.numpy as jnp
import numpy as np
import json
import equinox as eqx

import pickle
import matplotlib.pyplot as plt

# Load parameters from JSON file
with open("mini_wheelbot_default_parameters.json", "r") as f:
    params = json.load(f)


class PhysicsParameters(eqx.Module):
    """Trainable physics parameters for the wheelbot system."""
    m_WR: jnp.ndarray
    m_B: jnp.ndarray
    I_Wxz_Ryz: jnp.ndarray
    I_Wy_Rx: jnp.ndarray
    I_Bx: jnp.ndarray
    I_By: jnp.ndarray
    I_Bz: jnp.ndarray
    r_W: jnp.ndarray
    l_WB: jnp.ndarray
    fric_magn: jnp.ndarray
    fric_slope: jnp.ndarray
    kT_W: jnp.ndarray
    kT_R: jnp.ndarray
    
    def __init__(self, params_dict=None):
        """Initialize physics parameters from a dictionary or use defaults."""
        if params_dict is None:
            params_dict = params
        
        self.m_WR = jnp.array(params_dict["m_WR"])
        self.m_B = jnp.array(params_dict["m_B"])
        self.I_Wxz_Ryz = jnp.array(params_dict["I_Wxz_Ryz"])
        self.I_Wy_Rx = jnp.array(params_dict["I_Wy_Rx"])
        self.I_Bx = jnp.array(params_dict["I_Bx"])
        self.I_By = jnp.array(params_dict["I_By"])
        self.I_Bz = jnp.array(params_dict["I_Bz"])
        self.r_W = jnp.array(params_dict["r_W"])
        self.l_WB = jnp.array(params_dict["l_WB"] / 2)
        self.fric_magn = jnp.array(params_dict["fric_magn"])
        self.fric_slope = jnp.array(params_dict["fric_slope"])
        self.kT_W = jnp.array(1.0)
        self.kT_R = jnp.array(1.0)
    
    def get_derived_params(self):
        """Compute derived parameters (inertia matrices, etc.)."""
        g = jnp.array([0, 0, -9.81])
        r_R_B = jnp.array([0, 0, self.l_WB])
        r_W_B = jnp.array([0, 0, -self.l_WB])
        I_W = jnp.diag(jnp.array([self.I_Wxz_Ryz, self.I_Wy_Rx, self.I_Wxz_Ryz]))
        I_R = jnp.diag(jnp.array([self.I_Wy_Rx, self.I_Wxz_Ryz, self.I_Wxz_Ryz]))
        I_B = jnp.diag(jnp.array([self.I_Bx, self.I_By, self.I_Bz]))
        return g, r_R_B, r_W_B, I_W, I_R, I_B


# Default global parameters (for backward compatibility)
m_WR = params["m_WR"]
m_B = params["m_B"]
I_Wxz_Ryz = params["I_Wxz_Ryz"]
I_Wy_Rx = params["I_Wy_Rx"]
I_Bx = params["I_Bx"]
I_By = params["I_By"]
I_Bz = params["I_Bz"]
r_W = params["r_W"]
l_WB = params["l_WB"]/2
fric_magn = params["fric_magn"]
fric_slope = params["fric_slope"]
kT_W = 1
kT_R = 1

g = jnp.array([0,0,-9.81])
r_R_B = jnp.array([0,0,l_WB])
r_W_B = jnp.array([0,0,-l_WB])
I_W = jnp.diag(jnp.array([I_Wxz_Ryz, I_Wy_Rx, I_Wxz_Ryz]))
I_R = jnp.diag(jnp.array([I_Wy_Rx, I_Wxz_Ryz, I_Wxz_Ryz]))
I_B = jnp.diag(jnp.array([I_Bx, I_By, I_Bz]))


def rotate_vector_by_quaternion(q, v):
    """
    Rotate a vector by a quaternion.
    
    Args:
        q: quaternion in w-x-y-z notation (4-element array)
        v: vector in x-y-z notation (3-element array)
    
    Returns:
        rotated vector (3-element array)
    """
    qw, qx, qy, qz = q
    vx, vy, vz = v
    
    # Rotate vector using quaternion formula: v' = q * v * q^-1
    # Optimized formula: v' = v + 2 * cross(q_xyz, cross(q_xyz, v) + qw * v)
    q_xyz = jnp.array([qx, qy, qz])
    t = 2 * jnp.cross(q_xyz, jnp.cross(q_xyz, v) + qw * v)
    
    return v + t

def quat_to_rot(q):
    """ Convert quaternion w,x,y,z → rotation matrix. """
    qw, qx, qy, qz = q
    R = jnp.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz),   1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),   1-2*(qx**2+qy**2)]
    ])
    return R


def skew(w):
    """ Skew-symmetric matrix for cross product. """
    wx, wy, wz = w
    return jnp.array([
        [0, -wz, wy],
        [wz, 0, -wx],
        [-wy, wx, 0]
    ])


def compute_contact_point(r_B, q_B, physics_params=None):
    r_W_B_val = physics_params.get_derived_params()[2] if physics_params is not None else r_W_B
    r_W_val = physics_params.r_W if physics_params is not None else r_W
    r_W_I = r_B + rotate_vector_by_quaternion(q_B, r_W_B_val)
    
    # Get the wheel's y-axis in global frame (axis of rotation)
    By_axis_I = rotate_vector_by_quaternion(q_B, jnp.array([0, 1, 0]))

    # Get x-axis perpendicular to y_axis in x-y plane
    # Cross product of global z-axis with y_axis gives x-axis
    Iz_axis_I = jnp.array([0, 0, 1])
    Wx_axis_I = jnp.cross(By_axis_I,Iz_axis_I)
    Wx_axis_I = Wx_axis_I / jnp.linalg.norm(Wx_axis_I)

    # Calculate z-axis to complete right-hand coordinate system
    Wz_axis_I = jnp.cross(Wx_axis_I, By_axis_I)
    Wz_axis_I = Wz_axis_I / jnp.linalg.norm(Wz_axis_I)

    # Contact point is wheel center minus wheel radius in z direction
    r_contact_I = r_W_I - r_W_val * Wz_axis_I
    
    return r_contact_I

def contact_jacobian_A(r_B, q_B, physics_params=None):
    r_W_B_val = physics_params.get_derived_params()[2] if physics_params is not None else r_W_B
    r_W_val = physics_params.r_W if physics_params is not None else r_W
    
    # Axes in world
    By = rotate_vector_by_quaternion(q_B, jnp.array([0.0, 1.0, 0.0]))
    Iz = jnp.array([0.0, 0.0, 1.0])

    Wx = jnp.cross(By, Iz)
    Wx /= jnp.linalg.norm(Wx)

    Wz = jnp.cross(Wx, By)
    Wz /= jnp.linalg.norm(Wz)

    # Lateral direction in ground plane (orthogonal to Wx)
    Wlat = jnp.cross(Iz, Wx)
    Wlat /= jnp.linalg.norm(Wlat)

    # Contact point
    r_W_I = r_B + rotate_vector_by_quaternion(q_B, r_W_B_val)
    r_C_I = r_W_I - r_W_val * Wz
    d = r_C_I - r_B   # vector from body COM to contact

    # Build J_omega_B: mapping ω_B (in body frame) → contact velocity in world
    # Use basis e1,e2,e3 in body; ω_world = R_BI @ e_i
    e1 = jnp.array([1.0, 0.0, 0.0])
    e2 = jnp.array([0.0, 1.0, 0.0])
    e3 = jnp.array([0.0, 0.0, 1.0])

    R_BI = jnp.zeros((3,3))
    R_BI = R_BI.at[:,0].set(rotate_vector_by_quaternion(q_B, e1))
    R_BI = R_BI.at[:,1].set(rotate_vector_by_quaternion(q_B, e2))
    R_BI = R_BI.at[:,2].set(rotate_vector_by_quaternion(q_B, e3))

    J_omega = jnp.zeros((3,3))
    for i in range(3):
        w_world = R_BI[:,i]
        J_omega = J_omega.at[:, i].set(jnp.cross(w_world, d))

    # Assemble A (3x8) for [v_B(3), ω_B(3), dot_q_R, dot_q_W]
    A = jnp.zeros((3, 8))

    # Row 1: no motion along Iz
    A = A.at[0, 0:3].set(Iz)
    A = A.at[0, 3:6].set(Iz @ J_omega)

    # Row 2: no lateral slip
    A = A.at[1, 0:3].set(Wlat)
    A = A.at[1, 3:6].set(Wlat @ J_omega)

    # Row 3: rolling constraint
    A = A.at[2, 0:3].set(Wx)
    A = A.at[2, 3:6].set(Wx @ J_omega)
    A = A.at[2, 7].set(-r_W_val)   # derivative of (-r_W * dot_q_W)

    return A


def rnea_3link(x, qdd, use_gravity=True, physics_params=None):
    """
    Tailored inverse dynamics (RNEA-style) for the 3-body model.
    Returns generalized forces Q (8,) such that:
        Q = M(q) qdd + h(q,qdot)
    where h = C + G if use_gravity=True.
    
    Generalized velocities and accelerations:
        v = [v_B(3), omega_B(3), dot_q_R, dot_q_W]
        qdd = [a_B(3), alpha_B(3), ddq_R, ddq_W]
    All in WORLD frame.
    """

    # Get physics parameters
    if physics_params is not None:
        _, r_R_B_val, r_W_B_val, I_W_val, I_R_val, I_B_val = physics_params.get_derived_params()
        m_WR_val = physics_params.m_WR
        m_B_val = physics_params.m_B
    else:
        r_R_B_val, r_W_B_val = r_R_B, r_W_B
        I_W_val, I_R_val, I_B_val = I_W, I_R, I_B
        m_WR_val, m_B_val = m_WR, m_B

    # Unpack state
    r_B     = x[0:3]
    q_B     = x[3:7]
    q_R     = x[7]
    q_W     = x[8]
    v_B     = x[9:12]
    omega_B = x[12:15]    # world frame
    dot_q_R = x[15]
    dot_q_W = x[16]

    # Unpack accelerations
    a_B     = qdd[0:3]    # linear accel of body COM (world)
    alpha_B = qdd[3:6]    # angular accel of body (world)
    ddq_R   = qdd[6]
    ddq_W   = qdd[7]

    # Rotation body->world
    R_BI = quat_to_rot(q_B)

    # Vectors from body COM to link COMs in world
    r_R_rel = R_BI @ r_R_B_val
    r_W_rel = R_BI @ r_W_B_val

    r_R_I = r_B + r_R_rel
    r_W_I = r_B + r_W_rel

    # Joint axes in world (body x and y)
    e_x = jnp.array([1.0, 0.0, 0.0])
    e_y = jnp.array([0.0, 1.0, 0.0])
    axis_R = R_BI @ e_x    # reaction wheel axis
    axis_W = R_BI @ e_y    # driving wheel spin axis

    # Link orientations (body->link, then to world)
    R_RI = R_BI
    R_WI = R_BI

    # Inertias in WORLD frame
    I_B_world = R_BI @ I_B_val @ R_BI.T
    I_R_world = R_RI @ I_R_val @ R_RI.T
    I_W_world = R_WI @ I_W_val @ R_WI.T

    # Angular velocities in world
    omega_B_w = omega_B
    omega_R = omega_B_w + dot_q_R * axis_R
    omega_W = omega_B_w + dot_q_W * axis_W

    # Angular accelerations in world
    alpha_R = alpha_B + ddq_R * axis_R + jnp.cross(omega_B_w, dot_q_R * axis_R)
    alpha_W = alpha_B + ddq_W * axis_W + jnp.cross(omega_B_w, dot_q_W * axis_W)

    # Linear accelerations of COMs (world)
    # Base COM at r_B
    a_B_c = a_B

    # Joint positions relative to body COM
    r_BR_vec = r_R_I - r_B
    r_BW_vec = r_W_I - r_B

    # Accel of joint points (since COMs at joints: a_c = a_joint)
    a_R_c = a_B + jnp.cross(alpha_B, r_BR_vec) + jnp.cross(omega_B_w, jnp.cross(omega_B_w, r_BR_vec))
    a_W_c = a_B + jnp.cross(alpha_B, r_BW_vec) + jnp.cross(omega_B_w, jnp.cross(omega_B_w, r_BW_vec))

    # Gravity (world)
    g_vec = jnp.array([0.0, 0.0, -9.81]) if use_gravity else jnp.zeros(3)

    # Inertial linear forces (include gravity as external if use_gravity)
    F_B = m_B_val  * (a_B_c - g_vec)
    F_R = m_WR_val * (a_R_c - g_vec)
    F_W = m_WR_val * (a_W_c - g_vec)

    # Inertial moments about COM in world
    N_B = I_B_world @ alpha_B + jnp.cross(omega_B_w, I_B_world @ omega_B_w)
    N_R = I_R_world @ alpha_R + jnp.cross(omega_R,   I_R_world @ omega_R)
    N_W = I_W_world @ alpha_W + jnp.cross(omega_W,   I_W_world @ omega_W)

    # Joint torques (revolute around axis, COM at joint ⇒ tau = axis · N)
    tau_R = axis_R @ N_R
    tau_W = axis_W @ N_W

    # Base wrench from all bodies, expressed at body COM
    F_base = F_B + F_R + F_W
    N_base = (
        N_B +
        (N_R + jnp.cross(r_BR_vec, F_R)) +
        (N_W + jnp.cross(r_BW_vec, F_W))
    )

    # Pack generalized forces Q
    Q = jnp.zeros(8)
    Q = Q.at[0:3].set(F_base)       # force on base COM (world)
    Q = Q.at[3:6].set(N_base)       # moment about base COM (world)
    Q = Q.at[6].set(tau_R)
    Q = Q.at[7].set(tau_W)

    return Q

def compute_M_C_G_S_A_rnea_3link(x, physics_params=None):
    """
    Compute M(q), C(q,qdot), G(q), S, A(q) using RNEA.
    x: full 17-dim state.
    physics_params: Optional PhysicsParameters object
    """
    # Extract current generalized velocity v = [v_B, omega_B, dot_q_R, dot_q_W]
    v_B     = x[9:12]
    omega_B = x[12:15]
    dot_q_R = x[15]
    dot_q_W = x[16]
    v = jnp.hstack([v_B, omega_B, dot_q_R, dot_q_W])

    dof = 8

    # --- Bias term h(q,qdot) = C + G: accelerations = 0, gravity ON
    qdd_zero = jnp.zeros(dof)
    h = rnea_3link(x, qdd_zero, use_gravity=True, physics_params=physics_params)

    # # --- Mass matrix M(q): velocities = 0, gravity OFF, unit accelerations
    x_inert = x.at[9:].set(0.0)  # zero velocities
    M = jnp.zeros((dof, dof))
    for i in range(dof):
        qdd = jnp.zeros(dof)
        qdd = qdd.at[i].set(1.0)
        col = rnea_3link(x_inert, qdd, use_gravity=False, physics_params=physics_params)
        M = M.at[:, i].set(col)

    # --- Actuation matrix S (same as before)
    S = jnp.zeros((dof, 2))
    S = S.at[6, 0].set(1.0)  # tau_R
    S = S.at[7, 1].set(1.0)  # tau_W

    # --- Contact Jacobian A(q): same A(q) you already use,
    #     depending only on pose (r_B, q_B)
    r_B = x[0:3]
    q_B = x[3:7]
    A = contact_jacobian_A(r_B, q_B, physics_params)

    return M, h, S, A

# Gains
Kpitch = jnp.array([-0.4, -0.04, -0.004, -0.003])
Kroll  = jnp.array([-1.3, -0.16, -8e-05, -0.0004])

# -------------------------------
# Quaternion ↔ roll, pitch, yaw
# -------------------------------
def quat_to_rpy(q):
    w, x, y, z = q
    # Using aerospace ZYX convention
    sinr = 2*(w*x + y*z)
    cosr = 1 - 2*(x*x + y*y)
    roll = jnp.arctan2(sinr, cosr)

    sinp = 2*(w*y - z*x)
    pitch = jnp.arcsin(jnp.clip(sinp, -1, 1))

    siny = 2*(w*z + x*y)
    cosy = 1 - 2*(y*y + z*z)
    yaw = jnp.arctan2(siny, cosy)

    return roll, pitch, yaw


def rpy_to_quat(roll, pitch, yaw):
    """Convert roll-pitch-yaw to quaternion (w, x, y, z)."""
    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return jnp.array([qw, qx, qy, qz])


# -----------------------------------------------------
# Compute control torques τ_R, τ_W using RPY feedback
# -----------------------------------------------------
def compute_control_torques(x):
    q_B = x[3:7]
    omega_B = x[12:15]
    q_R     = x[7]
    q_W     = x[8]
    dot_q_R = x[15]
    dot_q_W = x[16]

    roll, pitch, yaw = quat_to_rpy(q_B)

    roll_rate  = omega_B[0]     # body rate p
    pitch_rate = omega_B[1]     # body rate q

    # Errors for control vectors
    roll_err_vec  = jnp.array([roll,  roll_rate,  q_R, dot_q_R])
    pitch_err_vec = jnp.array([pitch, pitch_rate, q_W, dot_q_W])

    tau_R = -Kroll @ roll_err_vec
    tau_W = -Kpitch @ pitch_err_vec

    tau_R = jnp.clip(tau_R, -0.5, 0.5)
    tau_W = jnp.clip(tau_W, -0.5, 0.5)

    return tau_W, tau_R



# -----------------------------------------------------
# VERY SMALL EULER ODE INTEGRATOR FOR THE SYSTEM
# -----------------------------------------------------
def step_dynamics(x, dt, compute_M_C_G_S_A=compute_M_C_G_S_A_rnea_3link, physics_params=None):
    tau_W, tau_R = compute_control_torques(x)
    step_dynamics_with_action(x, jnp.array([tau_W, tau_R]), dt, compute_M_C_G_S_A=compute_M_C_G_S_A_rnea_3link, physics_params=None)


def step_dynamics_with_action(x, u, dt, compute_M_C_G_S_A=compute_M_C_G_S_A_rnea_3link, physics_params=None):
    """
    Step dynamics with provided action instead of using controller.
    
    Args:
        x: State vector (17,)
        dt: Time step
        action: Action vector [tau_R, tau_W] (2,)
        compute_M_C_G_S_A: Function to compute system matrices
        physics_params: Optional PhysicsParameters object
    
    Returns:
        x_new: New state
        lam: Contact forces
        tau: Applied torques (same as action)
    """
    # ---------------------
    # Unpack state
    # ---------------------
    r_B     = x[0:3]
    q_B     = x[3:7]
    q_R     = x[7]
    q_W     = x[8]
    v_B     = x[9:12]
    omega_B = x[12:15]
    dot_q_R = x[15]
    dot_q_W = x[16]

    # Generalized velocity vector (8×1):
    # [ v_B(3), ω_B(3), dot_q_R, dot_q_W ]
    v = jnp.hstack([v_B, omega_B, dot_q_R, dot_q_W])

    # Get physics parameters
    if physics_params is not None:
        kT_R_val = physics_params.kT_R
        kT_W_val = physics_params.kT_W
        fric_magn_val = physics_params.fric_magn
        fric_slope_val = physics_params.fric_slope
    else:
        kT_R_val = kT_R
        kT_W_val = kT_W
        fric_magn_val = fric_magn
        fric_slope_val = fric_slope

    # ---------------------
    # System matrices
    # ---------------------
    M, C, S, A = compute_M_C_G_S_A(x, physics_params)

    # Use provided action
    tau_W, tau_R = u[0], u[1]
    tau = jnp.array([kT_R_val*tau_R, kT_W_val*tau_W])

    # ----------------------
    # FRICTION FORCES
    # ----------------------
    Q_fric = jnp.zeros(8)
    Iz = jnp.array([0.0, 0.0, 1.0])
    omega_yaw = jnp.dot(Iz, omega_B)  # world-z component of base angular velocity
    k_yaw   = fric_magn_val
    s_yaw   = fric_slope_val
    tau_z_fric = -k_yaw * jnp.tanh(omega_yaw * s_yaw)
    M_yaw = tau_z_fric * Iz
    Q_fric = Q_fric.at[3:6].set(Q_fric[3:6] + M_yaw)

    # ---------------------
    # Compute b(q, v) = -d/dt(A) * v
    # Approximate dA/dt numerically:
    # Ȧ ≈ (A(q + dt*v) - A(q)) / dt
    # ---------------------
    x_predict = x
    x_predict = x_predict.at[0:3].set(x_predict[0:3] + dt * v_B)  # r_B
    # quaternion update approximation
    wx, wy, wz = omega_B
    Omega = jnp.array([[0, -wx, -wy, -wz],
                      [wx,  0,  wz, -wy],
                      [wy, -wz,  0,  wx],
                      [wz,  wy, -wx, 0]])
    q_B_pred = q_B + 0.5 * dt * (Omega @ q_B)
    q_B_pred = q_B_pred / jnp.linalg.norm(q_B_pred)
    x_predict = x_predict.at[3:7].set(q_B_pred)
    x_predict = x_predict.at[7].set(x_predict[7] + dt * dot_q_R)
    x_predict = x_predict.at[8].set(x_predict[8] + dt * dot_q_W)
    x_predict = x_predict.at[9:12].set(v_B)
    x_predict = x_predict.at[12:15].set(omega_B)

    # A(current)
    A_now = A

    # A at predicted pose
    A_next = contact_jacobian_A(x_predict[0:3], x_predict[3:7], physics_params)

    A_dot = (A_next - A_now) / dt
    b = -A_dot @ v

    # ---------------------
    # Build and solve KKT system
    #
    # [ M   -Aᵀ ] [ qdd ] = [ Sτ - C - G ]
    # [ A     0 ] [ λ   ]   [ b          ]
    #
    # ---------------------
    # Left side matrix
    KKT = jnp.block([
        [ M,        -A_now.T ],
        [ A_now,    jnp.zeros((A_now.shape[0], A_now.shape[0])) ]
    ])

    rhs = jnp.hstack([
        S @ tau - C + Q_fric,
        b
    ])

    sol = jnp.linalg.solve(KKT, rhs)
    qdd = sol[:8]      # accelerations
    lam = sol[8:]      # contact forces (can log these!)

    # ---------------------
    # Integrate state using semi-implicit Euler
    # ---------------------

    # Linear motion
    v_B_new = v_B + dt * qdd[0:3]
    r_B_new = r_B + dt * v_B_new

    # Angular acceleration is in world frame already
    alpha_B = qdd[3:6]
    omega_B_world = omega_B + dt * alpha_B

    # Convert world → body frame angular velocity?
    # No: here omega_B is stored in BODY frame, so convert:
    R_BI = quat_to_rot(q_B)
    omega_B_new = R_BI.T @ (R_BI @ omega_B + dt * alpha_B)

    # Quaternion update
    wx, wy, wz = omega_B_new
    Omega = jnp.array([
        [0, -wx, -wy, -wz],
        [wx,  0,  wz, -wy],
        [wy, -wz, 0,  wx],
        [wz,  wy, -wx, 0]
    ])
    q_B_new = q_B + 0.5 * dt * (Omega @ q_B)
    q_B_new /= jnp.linalg.norm(q_B_new)

    # Joint angles
    dot_q_R_new = dot_q_R + dt * qdd[6]
    dot_q_W_new = dot_q_W + dt * qdd[7]
    q_R_new = q_R + dt * dot_q_R_new
    q_W_new = q_W + dt * dot_q_W_new

    # ---------------------
    # Pack new state
    # ---------------------
    x_new = jnp.zeros_like(x)
    x_new = x_new.at[0:3].set(r_B_new)
    x_new = x_new.at[3:7].set(q_B_new)
    x_new = x_new.at[7].set(q_R_new)
    x_new = x_new.at[8].set(q_W_new)
    x_new = x_new.at[9:12].set(v_B_new)
    x_new = x_new.at[12:15].set(omega_B_new)
    x_new = x_new.at[15].set(dot_q_R_new)
    x_new = x_new.at[16].set(dot_q_W_new)

    return x_new, lam, tau


# -----------------------------------------------------
# RUN A SHORT SIMULATION
# -----------------------------------------------------
@jax.tree_util.Partial(jax.jit, static_argnames=['N', 'use_controller'])
def simulate_jit(x0, dt, N, actions=None, use_controller=True, physics_params=None):
    """
    JIT-compiled simulation using jax.lax.scan for the rollout.
    
    Args:
        x0: Initial state (17,)
        dt: Time step
        N: Number of steps
        actions: Optional array of actions (N, 2) [tau_R, tau_W]. If None, uses controller.
        use_controller: If True, computes actions from controller. If False, uses provided actions.
        physics_params: Optional PhysicsParameters object. If None, uses global parameters.
    """
    
    def scan_step(carry, _):
        """Single step for scan: takes state, returns (new_state, outputs)"""
        x = carry
        x_new, lam, tau = step_dynamics(x, dt, physics_params=physics_params)
        # Stack outputs: state, lambda, tau
        outputs = (x_new, lam, tau)
        return x_new, outputs
    
    # Run scan over N steps
    _, trajectory_data = jax.lax.scan(scan_step, x0, None, length=N)

    # trajectory_data is a tuple of (states, lambdas, taus)
    trajectory, lambda_log, tau_log = trajectory_data
    
    # Prepend initial state to trajectory
    trajectory = jnp.concatenate([x0[jnp.newaxis, :], trajectory], axis=0)
    
    return trajectory, lambda_log, tau_log

def reset_contact_height(x):
    contact_offset = compute_contact_point(x[0:3], x[3:7])
    x = x.at[2].set(x[2] - contact_offset[2])
    return x

def simulate(T=2.0, dt=0.001):
    """Wrapper for simulate_jit that sets up initial conditions."""
    x = jnp.zeros(17)
    x = x.at[0:3].set(jnp.array([0.0, 0.0, r_W + l_WB]))
    
    # Set initial roll and pitch angles (in radians)
    roll_init = 0.17   # ~5.7 degrees
    pitch_init = 0.17  # ~8.6 degrees
    yaw_init = 0.0
    
    # Convert RPY to quaternion (ZYX convention)
    cy = jnp.cos(yaw_init * 0.5)
    sy = jnp.sin(yaw_init * 0.5)
    cp = jnp.cos(pitch_init * 0.5)
    sp = jnp.sin(pitch_init * 0.5)
    cr = jnp.cos(roll_init * 0.5)
    sr = jnp.sin(roll_init * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    x = x.at[3:7].set(jnp.array([qw, qx, qy, qz]))
    x = x.at[7].set(0.0)
    x = x.at[8].set(0.0)
    x = x.at[9:].set(0.0)
    
    x = reset_contact_height(x)
    
    N = int(T/dt)
    
    # Call JIT-compiled simulation
    trajectory, lambda_log, tau_log = simulate_jit(x, dt, N)
    
    # Convert to numpy for plotting/output
    return np.array(trajectory), np.array(lambda_log), np.array(tau_log)


def rpy_to_quaternion(roll, pitch, yaw):
    """
    Convert roll-pitch-yaw to quaternion (w, x, y, z).
    
    Args:
        roll, pitch, yaw: Euler angles in radians
    
    Returns:
        Quaternion as (w, x, y, z)
    """
    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return jnp.array([qw, qx, qy, qz])


def quaternion_to_rpy(q):
    """
    Convert quaternion (w, x, y, z) to roll-pitch-yaw.
    
    Args:
        q: Quaternion as array [w, x, y, z]
    
    Returns:
        (roll, pitch, yaw) in radians
    """
    w, x, y, z = q
    
    # Roll (x-axis rotation)
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll = jnp.arctan2(sinr, cosr)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = jnp.arcsin(jnp.clip(sinp, -1, 1))
    
    # Yaw (z-axis rotation)
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    yaw = jnp.arctan2(siny, cosy)
    
    return roll, pitch, yaw


def state_dict_to_vector(state_dict, state_labels):
    """
    Convert state dictionary to 17-dim state vector for simulation.
    
    The dataset contains: roll, pitch, yaw_vel, roll_vel, pitch_vel, drive_wheel, reaction_wheel, etc.
    The simulator expects: r_B(3), q_B(4), q_R, q_W, v_B(3), omega_B(3), dot_q_R, dot_q_W
    
    Args:
        state_dict: Dictionary mapping state labels to values
        state_labels: List of state label names (used to construct dict)
    
    Returns:
        17-dim state vector
    """
    # Extract from dataset format using actual field names
    roll = state_dict.get('/q_yrp/roll', 0.0)
    pitch = state_dict.get('/q_yrp/pitch', 0.0)
    yaw = state_dict.get('/q_yrp/yaw', 0.0)  # Updated to extract yaw
    
    yaw_vel = state_dict.get('/dq_yrp/yaw_vel', 0.0)
    roll_vel = state_dict.get('/dq_yrp/roll_vel', 0.0)
    pitch_vel = state_dict.get('/dq_yrp/pitch_vel', 0.0)
    
    q_W = state_dict.get('/q_DR/drive_wheel', 0.0)  # drive wheel position
    q_R = state_dict.get('/q_DR/reaction_wheel', 0.0)  # reaction wheel position
    dq_W = state_dict.get('/dq_DR/drive_wheel', 0.0)  # drive wheel velocity
    dq_R = state_dict.get('/dq_DR/reaction_wheel', 0.0)  # reaction wheel velocity
    
    # Position (not in dataset, assume 0)
    x, y, z = 0.0, 0.0, 0.0
    
    # Velocity (not directly in dataset, derive or assume 0)
    vx, vy, vz = 0.0, 0.0, 0.0
    
    # Convert RPY to quaternion
    quat = rpy_to_quaternion(roll, pitch, yaw)
    
    # Build 17-dim state vector
    # [r_B(3), q_B(4), q_R, q_W, v_B(3), omega_B(3), dot_q_R, dot_q_W]
    state_vector = jnp.array([
        x, y, z,  # r_B
        quat[0], quat[1], quat[2], quat[3],  # q_B (w, x, y, z)
        q_R, q_W,  # joint angles
        vx, vy, vz,  # v_B
        roll_vel, pitch_vel, yaw_vel,  # omega_B
        dq_R, dq_W  # joint velocities
    ])
    
    state_vector = reset_contact_height(state_vector)
    
    return state_vector


def state_vector_to_dict(state_vector, state_labels):
    """
    Convert 17-dim state vector to dictionary in dataset format.
    
    Args:
        state_vector: 17-dim state vector from simulator
        state_labels: List of state label names in dataset
    
    Returns:
        Dictionary with values for each state label
    """
    # Unpack simulator state
    r_B = state_vector[0:3]
    q_B = state_vector[3:7]
    q_R = state_vector[7]
    q_W = state_vector[8]
    v_B = state_vector[9:12]
    omega_B = state_vector[12:15]
    dot_q_R = state_vector[15]
    dot_q_W = state_vector[16]
    
    # Convert quaternion to RPY
    roll, pitch, yaw = quaternion_to_rpy(q_B)
    
    # Build dictionary matching dataset format
    state_dict = {
        '/q_yrp/roll': roll,
        '/q_yrp/pitch': pitch,
        '/q_yrp/yaw': yaw,
        '/dq_yrp/yaw_vel': omega_B[2],
        '/dq_yrp/roll_vel': omega_B[0],
        '/dq_yrp/pitch_vel': omega_B[1],
        "/q_DR/drive_wheel": q_W,
        "/q_DR/reaction_wheel": q_R,
        '/dq_DR/drive_wheel': dot_q_W,
        '/dq_DR/reaction_wheel': dot_q_R,
    }
    
    # Extract only the labels present in the dataset
    result = jnp.array([state_dict.get(label, 0.0) for label in state_labels])
    return result


def rollout_physics_model(physics_params, initial_state, actions_seq, dt):
    """
    Roll out the physics model for multiple steps.
    
    Args:
        physics_params: PhysicsParameters object
        initial_state: Starting state (17,) in simulator format
        actions_seq: Sequence of actions (N, 2) [tau_R, tau_W]
        dt: Time step
    
    Returns:
        predicted_states: (N, 17) predicted states at each timestep in simulator format
    """
    # N = actions_seq.shape[0]
    
    def step_fn(state, action):
        next_state, _,_ = step_dynamics_with_action(state, action, dt, physics_params=physics_params)
        return next_state, next_state
    
    _, predicted_states = jax.lax.scan(step_fn, initial_state, actions_seq)
    return predicted_states


def evaluate_physics_model(physics_params, dataset_path, rollout_length=100, num_eval_trajectories=100, save_prefix="physics_eval"):
    """
    Evaluate the physics model on a dataset by rolling out predictions and comparing to ground truth.
    
    Args:
        physics_params: PhysicsParameters object to evaluate
        dataset_path: Path to the dataset pickle file
        rollout_length: Number of steps to roll out
        num_eval_trajectories: Number of random trajectories to use for evaluation (default: 100)
        save_prefix: Prefix for saved plots
        
    Returns:
        Mean squared error over the rollout
    """

    
    # Load dataset
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
    
    states = jnp.array(data["states"])[:, 0, :]  # Initial states (N, n_obs)
    actions = jnp.array(data["actions"])[:, :rollout_length, :]  # (N, T, 2)
    next_states = jnp.array(data["nextstates"])[:, :rollout_length, :]  # (N, T, n_obs)
    states_labels = data.get("states_labels", [f"state_{i}" for i in range(states.shape[1])])
    actions_labels = data.get("actions_labels", [f"action_{i}" for i in range(actions.shape[2])])
    dt = data.get("dt", 0.001)  # Time step from dataset
    
    print(f"\n{'='*60}")
    print(f"Physics Model Evaluation")
    print(f"{'='*60}")
    print(f"Dataset loaded: {states.shape[0]} trajectories")
    print(f"State dim: {states.shape[1]}, Action dim: {actions.shape[2]}")
    print(f"Trajectory length: {actions.shape[1]}")
    print(f"Time step dt: {dt}")
    print(f"State fields: {states_labels}")
    print(f"Action fields: {actions_labels}")
    
    # Select random subset of trajectories
    key = jax.random.PRNGKey(42)
    num_total_trajectories = states.shape[0]
    num_eval = min(num_eval_trajectories, num_total_trajectories)
    eval_indices = jax.random.choice(key, num_total_trajectories, shape=(num_eval,), replace=False)
    
    states_eval = states[eval_indices]
    actions_eval = actions[eval_indices]
    next_states_eval = next_states[eval_indices]
    
    print(f"\nUsing {num_eval} random trajectories for evaluation")
    
    # Convert initial states from dataset format to simulator format
    def convert_initial_state(state_dict_vec):
        state_dict = {label: state_dict_vec[i] for i, label in enumerate(states_labels)}
        return state_dict_to_vector(state_dict, states_labels)
    
    initial_states_sim = jax.vmap(convert_initial_state)(states_eval)
    
    # Roll out physics model for all trajectories
    print("\nRolling out physics model...")
    
    def rollout_single_trajectory(initial_state, action_seq):
        return rollout_physics_model(physics_params, initial_state, action_seq, dt)
    
    rollout_batch = jax.vmap(rollout_single_trajectory)
    predicted_trajectories_sim = rollout_batch(initial_states_sim, actions_eval)
    
    # Convert predicted trajectories back to dataset format
    def convert_trajectory(traj_sim):
        return jax.vmap(lambda s: state_vector_to_dict(s, states_labels))(traj_sim)
    
    predicted_trajectories = jax.vmap(convert_trajectory)(predicted_trajectories_sim)
    
    # Compute MSE
    mse = jnp.mean((predicted_trajectories - next_states_eval) ** 2)
    print(f"\n{rollout_length}-step Rollout MSE: {mse:.6f}")
    
    # Compute MSE over time
    mse_over_time = jnp.mean((predicted_trajectories - next_states_eval) ** 2, axis=(0, 2))
    
    # Select 5 random trajectories for plotting
    num_plot_trajectories = min(5, num_eval)
    plot_key = jax.random.PRNGKey(123)
    plot_indices = jax.random.choice(plot_key, num_eval, shape=(num_plot_trajectories,), replace=False)
    
    # Plot trajectory comparisons
    nx = len(states_labels)
    fig, axes = plt.subplots(nx, 1, figsize=(12, 3 * nx))
    if nx == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(range(num_plot_trajectories))
    
    for state_idx in range(nx):
        ax = axes[state_idx]
        state_label = states_labels[state_idx]
        
        # Collect all ground truth values for this state dimension to determine y-axis limits
        gt_values_for_state = []
        for traj_idx in plot_indices:
            gt_trajectory = next_states_eval[traj_idx, :, state_idx]
            gt_values_for_state.append(gt_trajectory)
        
        # Calculate y-axis limits based on ground truth data
        all_gt_values = jnp.concatenate(gt_values_for_state)
        max_abs_gt = jnp.max(jnp.abs(all_gt_values))
        y_limit = 1.1 * max_abs_gt if max_abs_gt > 0 else 1.0
        
        for i, traj_idx in enumerate(plot_indices):
            gt_trajectory = next_states_eval[traj_idx, :, state_idx]
            pred_trajectory = predicted_trajectories[traj_idx, :, state_idx]
            
            time_steps = jnp.arange(rollout_length)
            color = colors[i]
            ax.plot(time_steps, gt_trajectory, '-', alpha=0.7, color=color, label=f'GT Traj {traj_idx+1}')
            ax.plot(time_steps, pred_trajectory, '--', alpha=0.7, color=color, label=f'Pred Traj {traj_idx+1}')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel(state_label)
        ax.set_title(f'{state_label} - {rollout_length}-Step Rollout (Physics Model)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-y_limit, y_limit)
    
    plt.tight_layout()
    output_traj = f'plots/{save_prefix}_trajectories.pdf'
    plt.savefig(output_traj, bbox_inches='tight')
    plt.close()
    print(f"Trajectory plots saved to {output_traj}")
    
    # Plot MSE over time
    plt.figure(figsize=(10, 6))
    plt.plot(jnp.arange(rollout_length), mse_over_time, marker='o', markersize=3)
    plt.xlabel('Time Step')
    plt.ylabel('MSE')
    plt.title(f'Prediction Error Over {rollout_length}-Step Rollout (Physics Model)')
    plt.grid(True, alpha=0.3)
    output_mse = f'plots/{save_prefix}_mse_over_time.pdf'
    plt.savefig(output_mse)
    plt.close()
    print(f"MSE over time plot saved to {output_mse}")
    print(f"{'='*60}\n")
    
    return mse


if __name__ == "__main__":
    
    
    # Check if we should run evaluation
    physics_params = PhysicsParameters()
    mse = evaluate_physics_model(
        physics_params,
        "dataset/dataset_100_step.pkl",
        rollout_length=1000,
        num_eval_trajectories=100,
        save_prefix="physics_eval"
    )
    
    exit()
    
    # Otherwise run normal simulation
    traj, lam, tau = simulate(4, 0.001)
    print("Final state:", traj[-1])
    print("Number of steps:", traj.shape[0])
    
    # Create time array
    t = np.arange(traj.shape[0]) * 0.001
    
    # Convert quaternions to roll-pitch-yaw
    rpy_traj = np.zeros((traj.shape[0], 3))
    for i in range(traj.shape[0]):
        roll, pitch, yaw = quat_to_rpy(traj[i, 3:7])
        rpy_traj[i, :] = [roll, pitch, yaw]
    
    # Plot trajectory with subplots (now 21 plots: 3 position + 3 RPY + 2 joints + 3 vel + 3 omega + 2 joint vel + 3 lam + 2 tau)
    fig, axes = plt.subplots(21, 1, figsize=(10, 21*2))
    fig.suptitle('Wheelbot State Trajectory')
    
    state_labels = [
        'r_B_x [m]', 'r_B_y [m]', 'r_B_z [m]',
        'roll [rad]', 'pitch [rad]', 'yaw [rad]',
        'q_R [rad]', 'q_W [rad]',
        'v_B_x [m/s]', 'v_B_y [m/s]', 'v_B_z [m/s]',
        'ω_B_x [rad/s]', 'ω_B_y [rad/s]', 'ω_B_z [rad/s]',
        'q̇_R [rad/s]', 'q̇_W [rad/s]',
        'λ_1 [N]', 'λ_2 [N]', 'λ_3 [N]',
        'τ_R [Nm]', 'τ_W [Nm]'
    ]
    
    # Plot position
    for i in range(3):
        axes[i].plot(t, traj[:, i])
        axes[i].set_ylabel(state_labels[i])
        axes[i].grid(True)
    
    # Plot roll-pitch-yaw
    for i in range(3):
        axes[i+3].plot(t, rpy_traj[:, i])
        axes[i+3].set_ylabel(state_labels[i+3])
        axes[i+3].grid(True)
    
    # Plot joint angles
    for i in range(2):
        axes[i+6].plot(t, traj[:, i+7])
        axes[i+6].set_ylabel(state_labels[i+6])
        axes[i+6].grid(True)
    
    # Plot velocities
    for i in range(3):
        axes[i+8].plot(t, traj[:, i+9])
        axes[i+8].set_ylabel(state_labels[i+8])
        axes[i+8].grid(True)
    
    # Plot angular velocities
    for i in range(3):
        axes[i+11].plot(t, traj[:, i+12])
        axes[i+11].set_ylabel(state_labels[i+11])
        axes[i+11].grid(True)
    
    # Plot joint velocities
    for i in range(2):
        axes[i+14].plot(t, traj[:, i+15])
        axes[i+14].set_ylabel(state_labels[i+14])
        axes[i+14].grid(True)
    
    # Plot contact forces (lambda)
    for i in range(3):
        axes[i+16].plot(t[:-1], lam[:, i])
        axes[i+16].set_ylabel(state_labels[i+16])
        axes[i+16].grid(True)
    
    # Plot control torques
    for i in range(2):
        axes[i+19].plot(t[:-1], tau[:, i])
        axes[i+19].set_ylabel(state_labels[i+19])
        axes[i+19].grid(True)
    
    axes[-1].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.savefig("plots/ode.pdf")
    # plt.show()
