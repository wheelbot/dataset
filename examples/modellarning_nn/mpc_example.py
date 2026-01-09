import casadi as cs
import numpy as np
import json
from pathlib import Path
import pickle
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import matplotlib.pyplot as plt

# Import the CasADi symbolic type from the export module
from modellearning_export_to_casadi import casadi_sym, CASADI_SYM_TYPE


def create_acados_model_from_casadi(casadi_path, json_path) -> AcadosModel:
    """
    Create an AcadosModel with discrete dynamics from a saved CasADi function.
    
    The model combines:
    1. Learned neural network dynamics for internal states (roll, pitch, velocities, etc.)
    2. Kinematic integration for pose states (position_x, position_y, yaw, drive_wheel_ang)
    
    State vector (13 states):
        [0] position_x (m)
        [1] position_y (m)
        [2] yaw (rad)
        [3] roll (rad)
        [4] pitch (rad)
        [5] yaw_vel (rad/s)
        [6] roll_vel (rad/s)
        [7] pitch_vel (rad/s)
        [8] drive_wheel_vel (rad/s)
        [9] reaction_wheel_vel (rad/s)
        [10] drive_wheel_accel (rad/s²)
        [11] reaction_wheel_accel (rad/s²)
        [12] battery_voltage (V)
    
    Control input vector (3 setpoints):
        [0] velocity_setpoint (m/s)
        [1] roll_setpoint (rad)
        [2] pitch_setpoint (rad)
    
    The model includes a prestabilizing controller that converts setpoints to torques:
        - Roll controller: torque_reaction_wheel = K_roll @ [roll_err, roll_vel_err, reaction_wheel_vel_err]
          with K_roll = [1.3, 0.16, 0.0004]
        - Pitch controller: torque_drive_wheel = K_pitch @ [pitch_err, pitch_vel_err, drive_wheel_vel_err]
          with K_pitch = [0.4, 0.04, 0.003]
    
    Args:
        casadi_path: Path to the .casadi file containing the learned dynamics
        json_path: Path to the .json file containing normalization parameters
        
    Returns:
        AcadosModel configured with discrete dynamics
    """
    casadi_path = Path(casadi_path)
    json_path = Path(json_path)
    
    # Load the learned dynamics function
    learned_dynamics = cs.Function.load(str(casadi_path))
    
    # Load normalization parameters and metadata from JSON
    with open(json_path, 'r') as f:
        model_data = json.load(f)
    
    # Extract normalization parameters
    state_mean = np.array(model_data['normalization']['state_mean'])
    state_std = np.array(model_data['normalization']['state_std'])
    action_mean = np.array(model_data['normalization']['action_mean'])
    action_std = np.array(model_data['normalization']['action_std'])
    
    # Extract NN parameters
    nn_params = model_data['nn_parameters']
    
    # Robot parameters
    wheel_radius = 0.032  # m
    dt = 0.01  # s
    
    # Create AcadosModel
    model = AcadosModel()
    model.name = 'wheelbot_dynamics'
    
    # Define symbolic variables for current state (13 states)
    x = casadi_sym.sym('x', 13)
    
    # Define symbolic variables for control inputs (3 setpoints)
    u = casadi_sym.sym('u', 3)
    
    # Extract state components
    position_x = x[0]
    position_y = x[1]
    yaw = x[2]
    roll = x[3]
    pitch = x[4]
    yaw_vel = x[5]
    roll_vel = x[6]
    pitch_vel = x[7]
    drive_wheel_vel = x[8]
    reaction_wheel_vel = x[9]
    drive_wheel_accel = x[10]
    reaction_wheel_accel = x[11]
    battery_voltage = x[12]
    
    # Learned states (indices 3-12 correspond to the 10 states the NN was trained on)
    learned_states = x[3:13]  # roll, pitch, yaw_vel, roll_vel, pitch_vel, 
                               # drive_wheel_vel, reaction_wheel_vel,
                               # drive_wheel_accel, reaction_wheel_accel, battery_voltage
    
    # Control inputs (setpoints)
    velocity_setpoint = u[0]  # m/s
    roll_setpoint = u[1]      # rad
    pitch_setpoint = u[2]     # rad
    
    # Convert velocity setpoint to drive wheel velocity setpoint
    drive_wheel_vel_setpoint = velocity_setpoint / wheel_radius
    
    # Prestabilizing controller gains
    K_roll = [1.3, 0.16, 0.0004]  # for [roll, roll_vel, reaction_wheel_vel]
    K_pitch = [0.4, 0.04, 0.003]  # for [pitch, pitch_vel, drive_wheel_vel]
    
    # Compute error vectors
    roll_errors = cs.vertcat(
        roll - roll_setpoint,
        roll_vel - 0.0,  # setpoint is 0
        reaction_wheel_vel - 0.0  # setpoint is 0
    )
    
    pitch_errors = cs.vertcat(
        pitch - pitch_setpoint,
        pitch_vel - 0.0,  # setpoint is 0
        drive_wheel_vel - drive_wheel_vel_setpoint
    )
    
    # Convert gains to CasADi vectors
    K_roll_vec = cs.vertcat(*K_roll)
    K_pitch_vec = cs.vertcat(*K_pitch)
    
    # Compute torques from prestabilizing controller using dot product
    torque_reaction_wheel = -cs.dot(K_roll_vec, roll_errors)
    torque_drive_wheel = -cs.dot(K_pitch_vec, pitch_errors)
    
    # Actions to pass to learned dynamics
    actions = cs.vertcat(torque_drive_wheel, torque_reaction_wheel)
    
    # Create CasADi parameters for normalization
    state_mean_param = casadi_sym.sym('state_mean', 10)
    state_std_param = casadi_sym.sym('state_std', 10)
    action_mean_param = casadi_sym.sym('action_mean', 2)
    action_std_param = casadi_sym.sym('action_std', 2)
    
    # Create CasADi parameters for NN weights/biases
    nn_param_symbols = []
    nn_param_names = []
    for param_name in sorted(nn_params.keys()):
        param_array = np.array(nn_params[param_name])
        if param_array.ndim == 1:
            # Bias vector
            param_sym = casadi_sym.sym(param_name, param_array.shape[0])
        else:
            # Weight matrix
            param_sym = casadi_sym.sym(param_name, param_array.shape[0], param_array.shape[1])
        nn_param_symbols.append(param_sym)
        nn_param_names.append(param_name)
    
    # Call the learned dynamics function
    # The function signature is: (state, action, nn_params..., state_mean, state_std, action_mean, action_std)
    learned_fn_args = [learned_states, actions] + nn_param_symbols + [
        state_mean_param, state_std_param, action_mean_param, action_std_param
    ]
    next_learned_states = learned_dynamics(*learned_fn_args)
    
    # Kinematic integration for pose states
    # Get drive wheel velocity from learned states (index 5 in learned_states, which is index 9 in full state)
    drive_wheel_vel = learned_states[5]  # drive_wheel_vel
    
    # Linear velocity in body frame (forward direction)
    v_body = wheel_radius * drive_wheel_vel
    
    # Transform to global frame
    next_position_x = position_x + dt * v_body * cs.cos(yaw)
    next_position_y = position_y + dt * v_body * cs.sin(yaw)
    
    # Yaw integration (yaw_vel is at index 2 in learned_states, which is index 6 in full state)
    yaw_vel = learned_states[2]
    next_yaw = yaw + dt * yaw_vel
    
    # Combine all next states
    xf = cs.vertcat(
        next_position_x,
        next_position_y,
        next_yaw,
        next_learned_states
    )
    
    # Set model variables
    model.x = x
    model.u = u
    model.disc_dyn_expr = xf
    
    # Create parameter vector for Acados
    # Order: NN parameters, then normalization parameters
    p = cs.vvcat(nn_param_symbols + [state_mean_param, state_std_param, 
                   action_mean_param, action_std_param])
    model.p = p
    
    # Store parameter values for easy access
    model.p_values = []
    for param_name in nn_param_names:
        param_array = np.array(nn_params[param_name])
        model.p_values.append(param_array.flatten())
    
    # Add normalization parameters
    model.p_values.extend([
        state_mean,
        state_std,
        action_mean,
        action_std
    ])
    
    # Concatenate all parameter values into a single array
    model.p_values_flat = np.concatenate([p.flatten() for p in model.p_values])
    
    # Set dimensions
    model.dims = {
        'nx': 13,  # Number of states
        'nu': 3,   # Number of control inputs (setpoints)
        'np': model.p_values_flat.shape[0]  # Number of parameters
    }
    
    print(f"AcadosModel created: {model.name}")
    print(f"  CasADi symbolic type: {CASADI_SYM_TYPE}")
    print(f"  States: {model.dims['nx']}")
    print(f"  Control inputs: {model.dims['nu']} (velocity_setpoint, roll_setpoint, pitch_setpoint)")
    print(f"  Parameters: {model.dims['np']}")
    print(f"  Discrete time step: {dt} s")
    print(f"  Wheel radius: {wheel_radius} m")
    print(f"  Prestabilizing controller:")
    print(f"    Roll:  K = [1.3, 0.16, 0.0004]")
    print(f"    Pitch: K = [0.4, 0.04, 0.003]")
    
    return model


def setup_mpc(model: AcadosModel, N_horizon=100):
    """
    Setup MPC using Acados with the given model.
    
    Args:
        model: AcadosModel with discrete dynamics
        N_horizon: Number of shooting nodes (default: 100)
        
    Returns:
        AcadosOcpSolver configured for tracking MPC
    """
    # Create OCP
    ocp = AcadosOcp()
    ocp.model = model
    
    # Dimensions
    nx = model.dims['nx']  # 13 states
    nu = model.dims['nu']  # 3 control inputs
    np_params = model.dims['np']  # number of parameters
    
    # Discretization
    ocp.dims.N = N_horizon
    dt = 0.01  # 10ms time step
    ocp.solver_options.tf = N_horizon * dt
    
    # Cost function - tracking quadratic cost
    # Reference trajectory: [x_ref, y_ref, yaw_ref, drive_wheel_vel_ref, ...]
    # Other states should go to zero
    
    # State weights (diagonal of Q matrix)
    Q_diag = np.array([
        1.0,   # position_x - track reference
        1.0,   # position_y - track reference
        5.0,    # yaw - track reference
        10.0,    # roll - regulate to zero
        10.0,    # pitch - regulate to zero
        0.5,    # yaw_vel - regulate to zero
        1.0,    # roll_vel - regulate to zero
        1.0,    # pitch_vel - regulate to zero
        5.0,    # drive_wheel_vel - track reference
        0.01,    # reaction_wheel_vel - regulate to zero
        0.01,    # drive_wheel_accel - regulate to zero
        0.01,    # reaction_wheel_accel - regulate to zero
        0.0    # battery_voltage - regulate to zero (or nominal)
    ])
    Q = np.diag(Q_diag)
    
    # Control input weights (diagonal of R matrix)
    R_diag = np.array([
        0.1,   # velocity_setpoint
        1.0,   # roll_setpoint
        1.0    # pitch_setpoint
    ])
    R = np.diag(R_diag)
    
    # Set cost matrices
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    
    # Output selection matrices
    # For stage cost: y = [x; u]
    Vx = np.zeros((nx + nu, nx))
    Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx = Vx
    
    Vu = np.zeros((nx + nu, nu))
    Vu[nx:, :] = np.eye(nu)
    ocp.cost.Vu = Vu
    
    # Terminal cost (only on states)
    Vx_e = np.eye(nx)
    ocp.cost.Vx_e = Vx_e
    
    # Cost weights
    W = np.block([[Q, np.zeros((nx, nu))],
                  [np.zeros((nu, nx)), R]])
    ocp.cost.W = W
    ocp.cost.W_e = Q  # Terminal cost only on states
    
    # Reference (will be updated at each step)
    y_ref = np.zeros(nx + nu)
    ocp.cost.yref = y_ref
    ocp.cost.yref_e = np.zeros(nx)
    
    # Constraints on control inputs
    ocp.constraints.lbu = np.array([-0.5, -0.3, -0.3])  # [vel, roll, pitch] lower bounds
    ocp.constraints.ubu = np.array([0.5, 0.3, 0.3])     # [vel, roll, pitch] upper bounds
    ocp.constraints.idxbu = np.array([0, 1, 2])
    
    # Initial state constraint
    ocp.constraints.x0 = np.zeros(nx)
    
    # Parameters (NN weights and normalization params)
    ocp.parameter_values = model.p_values_flat
    
    # Solver options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_OSQP'
    ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.regularize_method = 'MIRROR'
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.nlp_solver_max_iter = 1
    ocp.solver_options.globalization = 'FIXED_STEP'
    
    # Create solver
    solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    
    print(f"\nMPC configured:")
    print(f"  Horizon: {N_horizon} steps ({ocp.solver_options.tf:.2f}s)")
    print(f"  QP solver: OSQP (max iter: 50)")
    print(f"  NLP solver: SQP_RTI")
    print(f"  Globalization: FIXED_STEP")
    print(f"  Hessian approximation: Gauss-Newton")
    
    return solver


def generate_s_curve_reference(t_sim, velocity=0.2):
    """
    Generate an S-curve reference trajectory with trapezoidal velocity profile.
    
    The velocity profile:
    - Accelerates from 0 to target velocity with 0.2 m/s²
    - Maintains constant velocity in the middle
    - Decelerates to 0 with -0.2 m/s² at the end
    
    Args:
        t_sim: Array of simulation time steps
        velocity: Desired target velocity (m/s)
        
    Returns:
        Dictionary with reference arrays for x, y, yaw, velocity
    """
    # S-curve parameters
    straight_length = 1.0  # m
    curve_radius = 0.5     # m
    
    # Velocity profile parameters
    accel = 0.2  # m/s²
    t_accel = velocity / accel  # Time to reach target velocity
    t_decel = velocity / accel  # Time to decelerate to zero
    
    # Compute total path length (for the S-curve)
    total_path_length = 2 * straight_length + 2 * np.pi * curve_radius
    
    # Distance covered during acceleration and deceleration
    s_accel = 0.5 * accel * t_accel**2
    s_decel = 0.5 * accel * t_decel**2
    
    # Compute velocity and arc length for each time step
    v_ref = np.zeros_like(t_sim)
    s = np.zeros_like(t_sim)
    
    for i, t in enumerate(t_sim):
        if t < t_accel:
            # Acceleration phase
            v_ref[i] = accel * t
            s[i] = 0.5 * accel * t**2
        elif t < t_sim[-1] - t_decel:
            # Constant velocity phase
            v_ref[i] = velocity
            s[i] = s_accel + velocity * (t - t_accel)
        else:
            # Deceleration phase
            t_in_decel = t - (t_sim[-1] - t_decel)
            v_ref[i] = velocity - accel * t_in_decel
            s_at_decel_start = s_accel + velocity * (t_sim[-1] - t_decel - t_accel)
            s[i] = s_at_decel_start + velocity * t_in_decel - 0.5 * accel * t_in_decel**2
    
    # Generate position reference based on arc length
    x_ref = np.zeros_like(s)
    y_ref = np.zeros_like(s)
    yaw_ref = np.zeros_like(s)
    
    for i, s_i in enumerate(s):
        if s_i < straight_length:
            # First straight section
            x_ref[i] = s_i
            y_ref[i] = 0
            yaw_ref[i] = 0
            
        elif s_i < straight_length + np.pi * curve_radius:
            # First curve (right turn)
            s_curve = s_i - straight_length
            theta = s_curve / curve_radius
            x_ref[i] = straight_length + curve_radius * np.sin(theta)
            y_ref[i] = curve_radius * (1 - np.cos(theta))
            yaw_ref[i] = theta
            
        elif s_i < 2 * straight_length + np.pi * curve_radius:
            # Second straight section
            s_straight = s_i - straight_length - np.pi * curve_radius
            x_ref[i] = straight_length + s_straight
            y_ref[i] = 2 * curve_radius
            yaw_ref[i] = np.pi
            
        elif s_i < 2 * straight_length + 2 * np.pi * curve_radius:
            # Second curve (right turn, back to original heading)
            s_curve = s_i - 2 * straight_length - np.pi * curve_radius
            theta = s_curve / curve_radius
            x_ref[i] = straight_length - curve_radius * np.sin(theta)
            y_ref[i] = 2 * curve_radius + curve_radius * (1 - np.cos(theta))
            yaw_ref[i] = np.pi + theta
            
        else:
            # Final straight section
            s_straight = s_i - 2 * straight_length - 2 * np.pi * curve_radius
            x_ref[i] = -s_straight
            y_ref[i] = 2 * curve_radius
            yaw_ref[i] = 0
    
    return {
        'x': x_ref,
        'y': y_ref,
        'yaw': yaw_ref,
        'velocity': v_ref
    }


def simulate_mpc_closed_loop(solver, reference, x0, wheel_radius=0.032):
    """
    Run closed-loop MPC simulation.
    
    Args:
        solver: AcadosOcpSolver
        reference: Dict with reference trajectory (x, y, yaw, velocity)
        x0: Initial state
        wheel_radius: Robot wheel radius (m)
        
    Returns:
        Dictionary with simulation results
    """
    N_sim = len(reference['x'])
    N_horizon = solver.acados_ocp.dims.N
    nx = 13
    nu = 3
    
    # Storage for results
    x_sim = np.zeros((N_sim, nx))
    u_sim = np.zeros((N_sim, nu))
    x_pred = np.zeros((N_sim, N_horizon + 1, nx))  # Open-loop predictions
    
    # Initial state
    x_sim[0] = x0
    
    # Get the model and create a dynamics function for simulation
    model = solver.acados_ocp.model
    dynamics_fun = cs.Function('dynamics', [model.x, model.u, model.p], [model.disc_dyn_expr])
    
    # Get parameter values from the model
    p_values = model.p_values_flat
    
    print("\nRunning closed-loop simulation...")
    print(f"  Simulation steps: {N_sim}")
    print(f"  Using model discrete dynamics for propagation")
    
    for i in range(N_sim - 1):
        # Current state
        x_current = x_sim[i]
        
        # Set initial state constraint
        # solver.set(0, 'lbx', x_current)
        # solver.set(0, 'ubx', x_current)
        
        # Set reference for all shooting nodes
        for j in range(N_horizon):
            # Get reference at future time
            idx = min(i + j, N_sim - 1)
            
            # Convert velocity to drive wheel velocity
            drive_wheel_vel_ref = reference['velocity'][idx] / wheel_radius
            
            # State reference
            x_ref = np.zeros(nx)
            x_ref[0] = reference['x'][idx]      # position_x
            x_ref[1] = reference['y'][idx]      # position_y
            x_ref[2] = reference['yaw'][idx]    # yaw
            x_ref[8] = drive_wheel_vel_ref      # drive_wheel_vel
            # All other states: zero
            
            # Control reference (nominal: zero setpoints)
            u_ref = np.zeros(nu)
            
            y_ref = np.concatenate([x_ref, u_ref])
            solver.set(j, 'yref', y_ref)
        
        # Terminal reference
        idx = min(i + N_horizon, N_sim - 1)
        x_ref_e = np.zeros(nx)
        x_ref_e[0] = reference['x'][idx]
        x_ref_e[1] = reference['y'][idx]
        x_ref_e[2] = reference['yaw'][idx]
        x_ref_e[8] = reference['velocity'][idx] / wheel_radius
        solver.set(N_horizon, 'yref', x_ref_e)
        
        # Solve OCP
        # status = solver.solve()
        solver.solve_for_x0(x_current, fail_on_nonzero_status=False, print_stats_on_failure=False)
        status = solver.get_status()
        
        if status != 0:
            print(f"  Warning: Solver returned status {status} at step {i}")
        
        # Get optimal control
        u_opt = solver.get(0, 'u')
        u_sim[i] = u_opt
        
        # Store open-loop prediction
        for j in range(N_horizon + 1):
            x_pred[i, j] = solver.get(j, 'x')
        
        # Simulate system using the actual discrete dynamics
        x_next = dynamics_fun(x_current, u_opt, p_values).full().flatten()
        x_sim[i + 1] = x_next
        
        if i % 50 == 0:
            print(f"  Step {i}/{N_sim}: x={x_current[0]:.3f}, y={x_current[1]:.3f}, "
                  f"yaw={x_current[2]:.3f}")
    
    print("✓ Simulation complete!")
    
    return {
        'x_sim': x_sim,
        'u_sim': u_sim,
        'x_pred': x_pred
    }


def plot_results(results, reference):
    """
    Plot MPC results: trajectory tracking and state evolution.
    
    Args:
        results: Dictionary with simulation results
        reference: Dictionary with reference trajectory
    """
    x_sim = results['x_sim']
    x_pred = results['x_pred']
    N_sim = x_sim.shape[0]
    t_sim = np.arange(N_sim) * 0.01  # Time vector in seconds
    
    # Create figure with grid layout: state plots on left, x-y plot on right
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3, width_ratios=[1, 1])
    
    # State indices and labels for left column
    state_info = [
        (3, 'Roll [rad]'),
        (4, 'Pitch [rad]'),
        (6, 'Roll rate [rad/s]'),
        (7, 'Pitch rate [rad/s]'),
    ]
    
    # Create state subplots on the left side (4 rows)
    for idx, (state_idx, label) in enumerate(state_info):
        ax = fig.add_subplot(gs[idx, 0])
        
        # Plot closed-loop trajectory
        ax.plot(t_sim, x_sim[:, state_idx], 'b-', linewidth=2, label='Closed-loop', zorder=10)
        
        # Plot open-loop predictions with transparency
        for i in range(0, N_sim, 10):
            t_pred = t_sim[i] + np.arange(x_pred.shape[1]) * 0.01
            ax.plot(t_pred, x_pred[i, :, state_idx], 'r--', alpha=0.15, linewidth=1)
        
        # Reference line at zero (for regulation states)
        ax.axhline(0, color='k', linestyle=':', linewidth=1, alpha=0.5)
        
        ax.set_ylabel(label, fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Only show x-label on bottom plot
        if idx == len(state_info) - 1:
            ax.set_xlabel('Time [s]', fontsize=11)
        
        # Add legend to first subplot
        if idx == 0:
            ax.plot([], [], 'r--', alpha=0.3, linewidth=1, label='Open-loop pred.')
            ax.legend(fontsize=9, loc='upper right')
    
    # Create x-y trajectory plot on the right side (top half)
    ax_xy = fig.add_subplot(gs[0:2, 1])
    
    # Plot reference trajectory
    ax_xy.plot(reference['x'], reference['y'], 'k--', linewidth=2, label='Reference', zorder=5)
    
    # Plot closed-loop trajectory
    ax_xy.plot(x_sim[:, 0], x_sim[:, 1], 'b-', linewidth=2, label='Closed-loop', zorder=10)
    
    # Plot open-loop predictions with transparency
    for i in range(0, N_sim, 10):
        ax_xy.plot(x_pred[i, :, 0], x_pred[i, :, 1], 'r--', alpha=0.15, linewidth=1)
    
    # Add a dummy line for legend
    ax_xy.plot([], [], 'r--', alpha=0.3, linewidth=1, label='Open-loop pred.')
    
    # Mark start and end
    ax_xy.plot(x_sim[0, 0], x_sim[0, 1], 'go', markersize=10, label='Start', zorder=15)
    ax_xy.plot(x_sim[-1, 0], x_sim[-1, 1], 'rs', markersize=10, label='End', zorder=15)
    
    ax_xy.set_xlabel('x [m]', fontsize=12)
    ax_xy.set_ylabel('y [m]', fontsize=12)
    ax_xy.set_title('Trajectory Tracking', fontsize=12, fontweight='bold')
    ax_xy.legend(fontsize=9, loc='best')
    ax_xy.grid(True, alpha=0.3)
    ax_xy.axis('equal')
    
    # Create additional state plots on right side (bottom half)
    state_info_right = [
        (2, 'Yaw [rad]'),
        (5, 'Yaw rate [rad/s]'),
    ]
    
    for idx, (state_idx, label) in enumerate(state_info_right):
        ax = fig.add_subplot(gs[2 + idx, 1])
        
        # Plot closed-loop trajectory
        ax.plot(t_sim, x_sim[:, state_idx], 'b-', linewidth=2, label='Closed-loop', zorder=10)
        
        # Plot open-loop predictions with transparency
        for i in range(0, N_sim, 10):
            t_pred = t_sim[i] + np.arange(x_pred.shape[1]) * 0.01
            ax.plot(t_pred, x_pred[i, :, state_idx], 'r--', alpha=0.15, linewidth=1)
        
        # Reference line (yaw should track reference, yaw_rate should be zero)
        if state_idx == 2:  # yaw
            # Plot yaw reference
            ax.plot(t_sim, reference['yaw'], 'k:', linewidth=1.5, alpha=0.7, label='Reference')
        else:
            ax.axhline(0, color='k', linestyle=':', linewidth=1, alpha=0.5)
        
        ax.set_ylabel(label, fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Show x-label on bottom plot
        if idx == len(state_info_right) - 1:
            ax.set_xlabel('Time [s]', fontsize=11)
    
    # Add overall title
    fig.suptitle('MPC Trajectory Tracking: S-Curve', fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    filename = "plots/mpc_trajectory_tracking.pdf"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved as {filename}")
    plt.show()


if __name__ == "__main__":
    # Paths to exported model
    casadi_path = "models/trained_model_multistep.casadi"
    json_path = "models/trained_model_multistep.json"
    
    print("=" * 70)
    print("MPC Setup and Closed-Loop Simulation")
    print("=" * 70)
    
    # Create Acados model
    print("\n[1/4] Creating Acados model...")
    model = create_acados_model_from_casadi(casadi_path, json_path)
    
    # Setup MPC
    print("\n[2/4] Setting up MPC...")
    solver = setup_mpc(model, N_horizon=1)
    
    # Generate reference trajectory
    print("\n[3/4] Generating S-curve reference...")
    t_sim = np.arange(0, 0.1, 0.01)  # 2 seconds simulation
    reference = generate_s_curve_reference(t_sim, velocity=0.2)
    print(f"  Duration: {t_sim[-1]:.1f}s")
    print(f"  Velocity: 0.2 m/s")
    
    # Initial state (all zeros)
    x0 = np.zeros(13)
    
    # Run closed-loop simulation
    print("\n[4/4] Running closed-loop simulation...")
    results = simulate_mpc_closed_loop(solver, reference, x0)
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(results, reference)
    
    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)







