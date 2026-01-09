import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
import pickle
import matplotlib.pyplot as plt

from modellearning_eom import train_physics_params


# ------------------------------------------------------------
# Physics parameters (learned by gradient descent)
# ------------------------------------------------------------

class PhysicsParameters(eqx.Module):
    """
    Simple planar segway-like model.

    State: [phi, phi_dot, theta, theta_dot]
        phi         : body pitch angle, upright = 0
        phi_dot     : body pitch angular velocity
        theta       : drive wheel angle
        theta_dot   : drive wheel angular velocity

    Action: tau_cmd (motor current command ~ torque), we learn a torque_scale.
    
    Physical model:
        A segway is an inverted pendulum on wheels. The wheel torque:
        - Accelerates the wheel (and thus the robot forward/backward)
        - Creates a reaction torque on the body (tending to tip it)
        
        The ground contact constraint couples wheel motion to body motion:
        - Forward acceleration of the wheel contact point affects body dynamics
        - Body tilt affects wheel dynamics through the coupling
    """
    # Learnable parameters
    m_b: jax.Array = eqx.field(default_factory=lambda: jnp.array(0.5))          # body mass [kg]
    m_w: jax.Array = eqx.field(default_factory=lambda: jnp.array(0.1))          # wheel mass [kg]
    l: jax.Array = eqx.field(default_factory=lambda: jnp.array(0.06))         # CoM height above wheel axis [m]
    l_offset: jax.Array = eqx.field(default_factory=lambda: jnp.array(0.001)) # CoM horizontal offset [m]
    I_b: jax.Array = eqx.field(default_factory=lambda: jnp.array(300e-6))     # body inertia about wheel axis [kg m^2]
    I_w: jax.Array = eqx.field(default_factory=lambda: jnp.array(100e-6))     # wheel inertia [kg m^2]
    r_w: jax.Array = eqx.field(default_factory=lambda: jnp.array(0.032))       # wheel radius [m]
    b_phi: jax.Array = eqx.field(default_factory=lambda: jnp.array(0.001))    # body damping
    b_theta: jax.Array = eqx.field(default_factory=lambda: jnp.array(0.001))  # wheel damping
    torque_scale: jax.Array = eqx.field(default_factory=lambda: jnp.array(1.0))  # torque scale (Nm per unit command)

    def params_string(self) -> str:
        """Return a single-line string of all learned parameters."""
        return (f"m_b={float(self.m_b):.4f}, m_w={float(self.m_w):.4f}, l={float(self.l):.4f}, "
                f"l_offset={float(self.l_offset):.5f}, I_b={float(self.I_b):.6f}, "
                f"I_w={float(self.I_w):.6f}, r_w={float(self.r_w):.4f}, "
                f"b_phi={float(self.b_phi):.5f}, b_theta={float(self.b_theta):.6f}, "
                f"tau_s={float(self.torque_scale):.4f}")


# ------------------------------------------------------------
# State / action conversion (dataset ↔ simulator)
# ------------------------------------------------------------

def state_dict_to_vector(state_dict, state_labels):
    """
    Convert state dictionary (dataset format) to simulator state vector.

    Dataset fields:
        '/q_yrp/pitch'
        '/dq_yrp/pitch_vel'
        '/q_DR/drive_wheel'
        '/dq_DR/drive_wheel'

    Simulator state (4D):
        [phi, phi_dot, theta, theta_dot]
    """
    pitch = state_dict.get('/q_yrp/pitch', 0.0)
    pitch_vel = state_dict.get('/dq_yrp/pitch_vel', 0.0)
    q_dw = state_dict.get('/q_DR/drive_wheel', 0.0)
    dq_dw = state_dict.get('/dq_DR/drive_wheel', 0.0)

    state_vector = jnp.array([pitch, pitch_vel, q_dw, dq_dw])
    return state_vector


def state_vector_to_dict(state_vector, state_labels):
    """
    Convert simulator state vector to dataset-format observation vector.

    Simulator state:
        [phi, phi_dot, theta, theta_dot]

    Returns:
        jnp.array([...]) in the order of `state_labels`.
    """
    phi = state_vector[0]
    phi_dot = state_vector[1]
    theta = state_vector[2]
    theta_dot = state_vector[3]

    state_dict = {
        '/q_yrp/pitch': phi,
        '/dq_yrp/pitch_vel': phi_dot,
        '/q_DR/drive_wheel': theta,
        '/dq_DR/drive_wheel': theta_dot,
    }

    # Only return the labels present in the dataset, in the right order
    result = jnp.array([state_dict.get(label, 0.0) for label in state_labels])
    return result


# ------------------------------------------------------------
# Dynamics: segway-like inverted pendulum on wheels
# ------------------------------------------------------------


def _continuous_dynamics(params: PhysicsParameters, state, action):
    """
    Continuous-time dynamics x_dot = f(x, u) for the segway-like inverted pendulum.
    
    This models a simple wheeled inverted pendulum (segway) where:
    - The body pivots about the wheel axis
    - The wheel rolls on the ground (no slip assumed)
    - Motor torque acts between wheel and body
    
    State: (4,) [phi, phi_dot, theta, theta_dot]
        phi       : body pitch angle (0 = upright, positive = tilting forward)
        phi_dot   : body pitch angular velocity
        theta     : wheel angle
        theta_dot : wheel angular velocity
    
    Action: scalar or vector; first element is tau_cmd (motor torque command)
    
    Simplified dynamics (linearized for small angles):
        The key insight is that motor torque tau acts on the wheel, which:
        1. Accelerates the wheel: I_w * theta_ddot = tau - tau_friction
        2. Creates reaction on body: the body experiences -tau plus gravitational torque
        
        For a segway with wheel on ground, the equations couple through:
        - Ground reaction forces
        - The constraint that wheel contact point moves with the robot
        
    Simplified model (decoupled for basic system ID):
        Body: I_eff * phi_ddot = m*g*l*sin(phi) + m*g*l_offset*cos(phi) - tau - b_phi*phi_dot
        Wheel: I_w * theta_ddot = tau - b_theta*theta_dot
        
    where I_eff = I_b + m*l^2 is the effective body inertia about the wheel axis.
    
    Note: In a full segway model, there's coupling through the wheel-ground constraint.
    This simplified model treats them as mostly independent, which is reasonable for
    small motions around the upright equilibrium.
    """
    phi, phi_dot, theta, theta_dot = state

    # Make sure we can handle shape () or (1,) or (2,) actions
    tau_cmd = action[0]
    
    # Apply torque scaling (no BEMF clipping since we don't reach high velocities)
    tau = params.torque_scale * tau_cmd

    m_b = params.m_b
    m_w = params.m_w
    l = params.l
    l_offset = params.l_offset
    I_b = params.I_b
    I_w = params.I_w
    r_w = params.r_w
    b_phi = params.b_phi
    b_theta = params.b_theta
    g = 9.81

    # Effective inertias
    I_eff = I_b + m_b * l**2         # body about axle
    H = I_w + (m_b + m_w) * r_w**2         # wheel + translational reflected inertia

    # Useful coupling term
    C = m_b * l * r_w

    # Denominator for phi dynamics
    denom = I_eff - (C**2) / H

    # Compute phi
    phi_ddot = (
        m_b * g * (l * jnp.sin(phi) - l_offset * jnp.cos(phi))
        - b_phi * phi_dot
        + (C / H) * (- tau)
    ) / denom

    # Compute theta using phi
    theta_ddot = (tau - C * phi_ddot) / H

    return jnp.array([phi_dot, phi_ddot, theta_dot, theta_ddot])


def step_physics_model(params: PhysicsParameters, state, action, dt):
    """
    One integration step using RK4.
    """
    def f(s):
        return _continuous_dynamics(params, s, action)

    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)

    new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return new_state


def rollout_physics_model(params: PhysicsParameters, initial_state, actions, dt):
    """
    Roll out the physics model over a sequence of actions.

    Args:
        params: PhysicsParameters
        initial_state: (4,) simulator state at t=0
        actions: (T, A) action sequence; first column used as torque command
        dt: time step

    Returns:
        states: (T, 4) simulator states at times t=1..T (after each action step)
    """
    def step_fn(state, action_t):
        next_state = step_physics_model(params, state, action_t, dt)
        return next_state, next_state

    _, states = jax.lax.scan(step_fn, initial_state, actions)
    return states


def get_pitch_state_weights(state_labels):
    """
    Get state weights for the pitch model loss function.
    
    Weights are chosen based on physical reasoning:
    - phi (pitch angle): High weight (1.0) - primary state we care about for balance
    - phi_dot (pitch velocity): Medium-high weight (1e-2) - important for dynamics
    - theta (wheel angle): Very low weight (0) - absolute wheel angle not important (unbounded)
    - theta_dot (wheel velocity): Low weight (1e-6) - affects torque dynamics
    
    Args:
        state_labels: List of state label names from dataset
        
    Returns:
        jnp.array of weights in the order of state_labels
    """
    weight_map = {
        '/q_yrp/pitch': 1.0,            # Pitch angle - most important for balance
        '/dq_yrp/pitch_vel': 1.0e-2,    # Pitch velocity - important for dynamics
        '/q_DR/drive_wheel': 0,         # Wheel angle - less important (unbounded)
        '/dq_DR/drive_wheel': 1e-6,     # Wheel velocity - affects motor dynamics
    }
    
    weights = jnp.array([weight_map.get(label, 1.0) for label in state_labels])
    return weights


def evaluate_physics_model(physics_params, dataset_path, rollout_length=100, 
                           num_eval_trajectories=100, save_prefix="physics_eval_pitch"):
    """
    Evaluate the physics model on a dataset by rolling out predictions and comparing to ground truth.
    Uses the pitch-specific weighted loss for evaluation.
    
    Args:
        physics_params: PhysicsParameters object to evaluate
        dataset_path: Path to the dataset pickle file
        rollout_length: Number of steps to roll out
        num_eval_trajectories: Number of random trajectories to use for evaluation (default: 100)
        save_prefix: Prefix for saved plots
        
    Returns:
        Tuple of (weighted_mse, unweighted_mse)
    """
    
    # Load dataset
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
    
    states = jnp.array(data["states"])[:, 0, :]  # Initial states (N, n_obs)
    actions = jnp.array(data["actions"])[:, :rollout_length, :]  # (N, T, action_dim)
    next_states = jnp.array(data["nextstates"])[:, :rollout_length, :]  # (N, T, n_obs)
    states_labels = data.get("states_labels", [f"state_{i}" for i in range(states.shape[1])])
    actions_labels = data.get("actions_labels", [f"action_{i}" for i in range(actions.shape[2])])
    dt = data.get("dt", 0.001)  # Time step from dataset
    
    print(f"\n{'='*60}")
    print(f"Physics Model Evaluation (Pitch)")
    print(f"{'='*60}")
    print(f"Dataset loaded: {states.shape[0]} trajectories")
    print(f"State dim: {states.shape[1]}, Action dim: {actions.shape[2]}")
    print(f"Trajectory length: {actions.shape[1]}")
    print(f"Time step dt: {dt}")
    print(f"State fields: {states_labels}")
    print(f"Action fields: {actions_labels}")
    
    # Get state weights
    state_weights = get_pitch_state_weights(states_labels)
    print(f"\nState weights: {dict(zip(states_labels, state_weights))}")
    
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
    
    # Compute unweighted MSE
    unweighted_mse = jnp.mean((predicted_trajectories - next_states_eval) ** 2)
    
    # Compute weighted MSE
    squared_errors = (predicted_trajectories - next_states_eval) ** 2  # (N, T, n_obs)
    weights_reshaped = state_weights.reshape(1, 1, -1)  # (1, 1, n_obs)
    weighted_squared_errors = squared_errors * weights_reshaped
    weighted_mse = jnp.sum(weighted_squared_errors) / (num_eval * rollout_length * jnp.sum(state_weights))
    
    print(f"\n{rollout_length}-step Rollout Results:")
    print(f"  Unweighted MSE: {unweighted_mse:.6f}")
    print(f"  Weighted MSE:   {weighted_mse:.6f}")
    
    # Compute MSE over time (both weighted and unweighted)
    unweighted_mse_over_time = jnp.mean((predicted_trajectories - next_states_eval) ** 2, axis=(0, 2))
    weighted_mse_over_time = jnp.sum(weighted_squared_errors, axis=(0, 2)) / (num_eval * jnp.sum(state_weights))
    
    # Compute per-state MSE
    per_state_mse = jnp.mean((predicted_trajectories - next_states_eval) ** 2, axis=(0, 1))
    print(f"\nPer-state MSE:")
    for i, label in enumerate(states_labels):
        weighted_state_mse = per_state_mse[i] * state_weights[i]
        print(f"  {label}: unweighted: {per_state_mse[i]:.6f},  weighted: {weighted_state_mse:.6f} (weight: {state_weights[i]:.5f})")
    
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
        weight = state_weights[state_idx]
        
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
        ax.set_title(f'{state_label} (weight={weight:.1f}) - {rollout_length}-Step Rollout')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-y_limit, y_limit)
    
    plt.tight_layout()
    output_traj = f'plots/{save_prefix}_trajectories.pdf'
    plt.savefig(output_traj, bbox_inches='tight')
    plt.close()
    print(f"\nTrajectory plots saved to {output_traj}")
    
    # Plot MSE over time (both weighted and unweighted)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(jnp.arange(rollout_length), unweighted_mse_over_time, marker='o', markersize=3, label='Unweighted')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('MSE')
    ax1.set_title(f'Unweighted Prediction Error Over {rollout_length}-Step Rollout')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(jnp.arange(rollout_length), weighted_mse_over_time, marker='o', markersize=3, 
             color='orange', label='Weighted')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Weighted MSE')
    ax2.set_title(f'Weighted Prediction Error Over {rollout_length}-Step Rollout')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    output_mse = f'plots/{save_prefix}_mse_over_time.pdf'
    plt.savefig(output_mse)
    plt.close()
    print(f"MSE over time plot saved to {output_mse}")
    print(f"{'='*60}\n")
    
    return weighted_mse, unweighted_mse


if __name__ == "__main__":
    # Load dataset
    with open("dataset/pitch.pkl", "rb") as f:
        data = pickle.load(f)
    
    MAX_ROLLOUT_LENGTH = 100
    
    states = jnp.array(data["states"])[:, 0, :]  # Initial states (N, n_obs)
    actions = jnp.array(data["actions"])[:, :MAX_ROLLOUT_LENGTH, :]  # (N, T, action_dim)
    next_states = jnp.array(data["nextstates"])[:, :MAX_ROLLOUT_LENGTH, :]  # (N, T, n_obs)
    states_labels = data.get("states_labels", [f"state_{i}" for i in range(states.shape[1])])
    actions_labels = data.get("actions_labels", [f"action_{i}" for i in range(actions.shape[2])])
    dt = data.get("dt", 0.01)  # Time step from dataset
    
    print(f"Dataset loaded: {states.shape[0]} trajectories")
    print(f"State dim: {states.shape[1]}, Action dim: {actions.shape[2]}")
    print(f"Trajectory length: {actions.shape[1]}")
    print(f"Time step dt: {dt}")
    print(f"State fields: {states_labels}")
    print(f"Action fields: {actions_labels}")
    print()
    
    # Initialize physics parameters
    physics_params_init = PhysicsParameters()
    
    # Get state weights for pitch-specific loss
    state_weights = get_pitch_state_weights(states_labels)
    print(f"State weights: {dict(zip(states_labels, state_weights))}")
    print()
    
    # Evaluate initial/default parameters before training
    print("\n" + "="*60)
    print("EVALUATING INITIAL/DEFAULT PARAMETERS")
    print("="*60)
    print(f"Initial Parameters: {physics_params_init.params_string()}")
    evaluate_physics_model(
        physics_params=physics_params_init,
        dataset_path="dataset/pitch.pkl",
        rollout_length=MAX_ROLLOUT_LENGTH,
        num_eval_trajectories=100,
        save_prefix="physics_eval_pitch_init"
    )
    
    # Train physics parameters using the generic training function
    physics_params, train_losses, epoch_numbers, curriculum_schedule = train_physics_params(
        physics_params_init=physics_params_init,
        states_initial=states,
        actions_trajectories=actions,
        states_targets=next_states,
        state_labels=states_labels,
        dt=dt,
        rollout_fn=rollout_physics_model,
        state_to_obs_fn=state_vector_to_dict,
        state_dict_to_vector_fn=state_dict_to_vector,
        state_weights=state_weights,
        min_rollout_length=30,
        max_rollout_length=MAX_ROLLOUT_LENGTH,
        num_epochs=300,
        batch_size=1024,
        learning_rate=1e-4
    )
    
    # Print learned parameters
    print(f"\nLearned Parameters: {physics_params.params_string()}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(epoch_numbers, train_losses, label='Training Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Physics Parameter Learning Loss (Pitch Model)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_yscale('log')
    
    ax2.plot(range(1, len(curriculum_schedule) + 1), curriculum_schedule, 
             marker='o', markersize=3, linewidth=2, color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Rollout Length')
    ax2.set_title('Curriculum Schedule')
    ax2.grid(True)
    ax2.set_ylim([0, MAX_ROLLOUT_LENGTH + 1])
    
    plt.tight_layout()
    plt.savefig('plots/loss_physics_learning_pitch.pdf')
    plt.close()
    print("\nLoss plot saved to plots/loss_physics_learning_pitch.pdf")
    
    # Save learned parameters
    with open("models/learned_physics_params_pitch.pkl", "wb") as f:
        pickle.dump({
            'params': physics_params,
            'losses': train_losses,
            'curriculum': curriculum_schedule
        }, f)
    print("Learned parameters saved to models/learned_physics_params_pitch.pkl")
    
    # Evaluate the trained model
    evaluate_physics_model(
        physics_params=physics_params,
        dataset_path="dataset/pitch.pkl",
        rollout_length=MAX_ROLLOUT_LENGTH,
        num_eval_trajectories=100,
        save_prefix="physics_eval_pitch"
    )
