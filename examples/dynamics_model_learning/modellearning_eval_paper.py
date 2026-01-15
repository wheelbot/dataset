"""
Evaluation script for dynamics models on long rollouts.

Tests both one-step and multi-step trained models on extended rollout sequences
to compare their long-term prediction accuracy.
"""

import jax
import pickle
from modellearning_common import load_dynamics_model
from modellearning_multistep_paper import rollout_model
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import rc


def evaluate_rollout(model_path, dataset_path, rollout_length=100, save_prefix="rollout"):
    """
    Evaluate a trained dynamics model on long rollouts.
    
    Args:
        model_path: Path to the saved model file
        dataset_path: Path to the rollout dataset
        rollout_length: Number of steps to roll out
        save_prefix: Prefix for saved plots
        
    Returns:
        Mean squared error over the rollout
    """
    
    model, hyperparams = load_dynamics_model(model_path)
    
    state_mean = jnp.array(hyperparams["state_mean"])
    state_std = jnp.array(hyperparams["state_std"])
    action_mean = jnp.array(hyperparams["action_mean"])
    action_std = jnp.array(hyperparams["action_std"])
    state_dim = hyperparams["state_dim"]
    training_type = hyperparams.get("training_type", "onestep")
    model_states_labels = hyperparams.get("states_labels", None)
    model_actions_labels = hyperparams.get("actions_labels", None)
    
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
    
    test_data = data["test"]
    states = jnp.squeeze(jnp.array(test_data["states"]))
    actions = jnp.array(test_data["actions"])[:, :rollout_length, :]
    next_states = jnp.array(test_data["nextstates"])[:, :rollout_length, :]
    dataset_states_labels = data.get("states_labels", None)
    dataset_actions_labels = data.get("actions_labels", None)
    
    # Verify that model and dataset have compatible fields
    if model_states_labels is not None and dataset_states_labels is not None:
        assert model_states_labels == dataset_states_labels, (
            f"State field mismatch!\n"
            f"Model trained on: {model_states_labels}\n"
            f"Dataset contains: {dataset_states_labels}"
        )
    
    if model_actions_labels is not None and dataset_actions_labels is not None:
        assert model_actions_labels == dataset_actions_labels, (
            f"Action field mismatch!\n"
            f"Model trained on: {model_actions_labels}\n"
            f"Dataset contains: {dataset_actions_labels}"
        )
    
    D = states.shape[0]
    N_sim = actions.shape[1]
    
    print(f"\nEvaluating: {model_path}")
    print(f"Training type: {training_type}")
    print(f"Dataset: {D} trajectories, {N_sim} steps")
    print(f"State dim: {state_dim}, Action dim: {actions.shape[2]}")
    if model_states_labels is not None:
        print(f"State fields: {model_states_labels}")
    if model_actions_labels is not None:
        print(f"Action fields: {model_actions_labels}")
    print(f"✓ Fields verified: model and dataset are compatible")
    
    normalized_states = (states - state_mean) / state_std
    normalized_actions = (actions - action_mean) / action_std
    normalized_next_states = (next_states - state_mean) / state_std
    
    rollout_batch = jax.vmap(rollout_model, in_axes=(None, 0, 0))
    predicted_trajectories = rollout_batch(model, normalized_states, normalized_actions)
    
    mse = jnp.mean((predicted_trajectories - normalized_next_states) ** 2)
    print(f"{N_sim}-step Rollout MSE: {mse:.6f}")
    
    predicted_trajectories_denorm = predicted_trajectories * state_std + state_mean
    
    num_plot_trajectories = min(5, D)
    nx = state_dim
    
    # Select 5 random trajectory indices
    key = jax.random.PRNGKey(42)
    random_indices = jax.random.choice(key, D, shape=(num_plot_trajectories,), replace=False)
    
    fig, axes = plt.subplots(nx, 1, figsize=(12, 3 * nx))
    if nx == 1:
        axes = [axes]
    
    for state_idx in range(nx):
        ax = axes[state_idx]
        
        # Get state label if available
        state_label = model_states_labels[state_idx] if model_states_labels is not None else f'State {state_idx+1}'
        print(f"State label: {state_label}  State idx: {state_idx}")

        # Collect all ground truth values for this state dimension to determine y-axis limits
        gt_values_for_state = []
        for traj_idx in random_indices:
            gt_trajectory = next_states[traj_idx, :, state_idx]
            gt_values_for_state.append(gt_trajectory)
        
        # Calculate y-axis limits based on ground truth data
        all_gt_values = jnp.concatenate(gt_values_for_state)
        max_abs_gt = jnp.max(jnp.abs(all_gt_values))
        y_limit = 1.1 * max_abs_gt
        
        # Use a color cycle for matching GT and Pred trajectories
        colors = plt.cm.tab10(range(num_plot_trajectories))
        
        for i, traj_idx in enumerate(random_indices):
            gt_trajectory = next_states[traj_idx, :, state_idx]
            pred_trajectory = predicted_trajectories_denorm[traj_idx, :, state_idx]
            
            time_steps = jnp.arange(N_sim)
            color = colors[i]
            ax.plot(time_steps, gt_trajectory, '-', alpha=0.7, color=color, label=f'GT Traj {traj_idx+1}')
            ax.plot(time_steps, pred_trajectory, '--', alpha=0.7, color=color, label=f'Pred Traj {traj_idx+1}')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel(state_label)
        ax.set_title(f'{state_label} - {N_sim}-Step Rollout ({training_type})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-y_limit, y_limit)
    
    plt.tight_layout()
    output_traj = f'plots/{save_prefix}_trajectories_{training_type}.pdf'
    plt.savefig(output_traj, bbox_inches='tight')
    plt.close()
    print(f"Trajectory plots saved to {output_traj}")
    
    mse_over_time = jnp.mean((predicted_trajectories - normalized_next_states) ** 2, axis=(0, 2))
    
    plt.figure(figsize=(10, 6))
    plt.plot(jnp.arange(N_sim), mse_over_time, marker='o', markersize=3)
    plt.xlabel('Time Step')
    plt.ylabel('MSE')
    plt.title(f'Prediction Error Over {N_sim}-Step Rollout ({training_type})')
    plt.grid(True, alpha=0.3)
    output_mse = f'plots/{save_prefix}_mse_over_time_{training_type}.pdf'
    plt.savefig(output_mse)
    plt.close()
    print(f"MSE over time plot saved to {output_mse}")
    
    return mse


def plot_paper_comparison(model_path, dataset_path, rollout_length=50):
    """
    Generate paper-ready 1x3 comparison plot for roll_vel, pitch_vel, and yaw_vel.
    """
    model, hyperparams = load_dynamics_model(model_path)
    
    state_mean = jnp.array(hyperparams["state_mean"])
    state_std = jnp.array(hyperparams["state_std"])
    action_mean = jnp.array(hyperparams["action_mean"])
    action_std = jnp.array(hyperparams["action_std"])
    model_states_labels = hyperparams.get("states_labels", None)
    
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
    
    test_data = data["test"]
    states = jnp.squeeze(jnp.array(test_data["states"]))
    actions = jnp.array(test_data["actions"])[:, :rollout_length, :]
    next_states = jnp.array(test_data["nextstates"])[:, :rollout_length, :]
    
    # Run rollout on first trajectory of test set
    traj_idx = 149 + 1
    normalized_state = (states[traj_idx] - state_mean) / state_std
    normalized_actions = (actions[traj_idx] - action_mean) / action_std
    
    predicted_traj_norm = rollout_model(model, normalized_state, normalized_actions)
    predicted_traj = predicted_traj_norm * state_std + state_mean
    gt_traj = next_states[traj_idx]
    
    # State indices
    # 2: /dq_yrp/yaw_vel
    # 3: /dq_yrp/roll_vel
    # 4: /dq_yrp/pitch_vel
    state_indices = [4, 5, 3]
    state_names = ["roll_vel", "pitch_vel", "yaw_vel"]
    bg_colors = ['#FFDFC3', '#E5F3E5', '#E3EEF5']
    
    # Plot settings
    width = 3.4125
    height = 0.8
    rc('font', size=8)
    rc('text', usetex=False)
    title_fontsize = 6
    tick_fontsize = 5
    linewidth = 0.8
    
    fig, axes = plt.subplots(1, 3, figsize=(width, height))
    time_steps = jnp.arange(rollout_length) * 0.01 # Assuming 100Hz
    
    for i, idx in enumerate(state_indices):
        ax = axes[i]
        ax.plot(time_steps, predicted_traj[:, idx], label='Model', linewidth=linewidth)
        ax.plot(time_steps, gt_traj[:, idx], label='MD', linewidth=linewidth, linestyle='--', alpha=0.7)
        
        if i == 0:
            ax.set_ylabel('Vel [rad/s]', fontsize=tick_fontsize, labelpad=1)
        
        ax.text(0.5, 0.9, state_names[i], transform=ax.transAxes, fontsize=title_fontsize, 
                ha='center', va='top', bbox=dict(facecolor=bg_colors[i], alpha=1, edgecolor='none', pad=1))
        
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, pad=0.5)
        ax.grid(True, linewidth=0.3, alpha=0.6, linestyle='--')
        ax.set_xlabel('Time [s]', fontsize=tick_fontsize, labelpad=1)

    # Legend in bottom right of Yaw plot
    handles, labels = axes[0].get_legend_handles_labels()
    axes[2].legend(handles, labels, loc='lower right', fontsize=5, frameon=True, labelspacing=0.1, handlelength=1.0)
    
    plt.tight_layout(pad=0.2)
    output_path = f"plots/modellearning_comparison_paper_{traj_idx}.pdf"
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"Paper plot saved to {output_path}")


def evaluate_feedback_control(model_path, rollout_length=500, save_prefix="feedback_control", dt=0.01):
    """
    Evaluate a trained dynamics model with linear feedback control from zero initial state.
    
    The controller stabilizes roll and pitch using linear feedback gains.
    
    State ordering in model:
        0: /q_yrp/roll
        1: /q_yrp/pitch
        2: /dq_yrp/yaw_vel
        3: /dq_yrp/roll_vel
        4: /dq_yrp/pitch_vel
        5: /dq_DR/drive_wheel
        6: /dq_DR/reaction_wheel
        7: /ddq_DR/drive_wheel
        8: /ddq_DR/reaction_wheel
        9: battery/voltage
    
    Args:
        model_path: Path to the saved model file
        rollout_length: Number of steps to simulate
        save_prefix: Prefix for saved plots
        dt: Time step for integration (0.01s)
        
    Returns:
        Dictionary with trajectory data
    """
    
    # Load model
    model, hyperparams = load_dynamics_model(model_path)
    
    state_mean = jnp.array(hyperparams["state_mean"])
    state_std = jnp.array(hyperparams["state_std"])
    action_mean = jnp.array(hyperparams["action_mean"])
    action_std = jnp.array(hyperparams["action_std"])
    state_dim = hyperparams["state_dim"]
    training_type = hyperparams.get("training_type", "onestep")
    model_states_labels = hyperparams.get("states_labels", None)
    model_actions_labels = hyperparams.get("actions_labels", None)
    
    print(f"\nEvaluating with feedback control: {model_path}")
    print(f"Training type: {training_type}")
    print(f"Rollout length: {rollout_length} steps ({rollout_length * dt:.1f}s)")
    print(f"State dim: {state_dim}")
    if model_states_labels is not None:
        print(f"State fields: {model_states_labels}")
    if model_actions_labels is not None:
        print(f"Action fields: {model_actions_labels}")
    
    # Controller gains
    # Pitch: [pitch_angle, pitch_vel, drive_wheel_angle, drive_wheel_vel]
    K_pitch = -jnp.array([-0.4, -0.04, -0.004, -0.003])
    # Roll: [roll_angle, roll_vel, reaction_wheel_angle, reaction_wheel_vel]
    K_roll = -jnp.array([-1.3, -0.16, -8e-05, -0.0004])
    
    # State indices (based on the ordering provided)
    idx_roll = 0
    idx_pitch = 1
    idx_yaw_vel = 2
    idx_roll_vel = 3
    idx_pitch_vel = 4
    idx_drive_wheel_vel = 5
    idx_reaction_wheel_vel = 6
    # idx 7 and 8 are accelerations
    # idx 9 is battery voltage
    
    # Initialize state to zeros
    state = jnp.zeros(state_dim)
    
    # Storage for trajectory
    states_history = []
    actions_history = []
    drive_wheel_angles = []
    reaction_wheel_angles = []
    
    # Initialize wheel angles (not part of state, need to integrate)
    drive_wheel_angle = 0.0
    reaction_wheel_angle = 0.0
    
    # Simulation loop
    for step in range(rollout_length):
        states_history.append(state)
        drive_wheel_angles.append(drive_wheel_angle)
        reaction_wheel_angles.append(reaction_wheel_angle)
        
        # Compute control actions using linear feedback
        # Pitch control (action index 0): u_pitch = K_pitch @ [pitch, pitch_vel, drive_angle, drive_vel]
        u_pitch = (K_pitch[0] * state[idx_pitch] + 
                   K_pitch[1] * state[idx_pitch_vel] +
                   K_pitch[2] * drive_wheel_angle +
                   K_pitch[3] * state[idx_drive_wheel_vel])
        
        # Roll control (action index 1): u_roll = K_roll @ [roll, roll_vel, reaction_angle, reaction_vel]
        u_roll = (K_roll[0] * state[idx_roll] +
                  K_roll[1] * state[idx_roll_vel] +
                  K_roll[2] * reaction_wheel_angle +
                  K_roll[3] * state[idx_reaction_wheel_vel])
        
        action = jnp.clip(jnp.array([u_pitch, u_roll]), -0.5, 0.5)
        actions_history.append(action)
        
        # Normalize state and action for model prediction
        normalized_state = (state - state_mean) / state_std
        normalized_action = (action - action_mean) / action_std
        
        # Predict next state
        model_input = jnp.concatenate([normalized_state, normalized_action])
        delta_state = model(model_input)
        normalized_next_state = normalized_state + delta_state
        
        # Denormalize
        state = normalized_next_state * state_std + state_mean
        
        # Integrate wheel angles
        drive_wheel_angle += state[idx_drive_wheel_vel] * dt
        reaction_wheel_angle += state[idx_reaction_wheel_vel] * dt
    
    # Convert to arrays
    states_history = jnp.array(states_history)
    actions_history = jnp.array(actions_history)
    drive_wheel_angles = jnp.array(drive_wheel_angles)
    reaction_wheel_angles = jnp.array(reaction_wheel_angles)
    time_steps = jnp.arange(rollout_length) * dt
    
    print(f"\nRollout statistics:")
    print(f"  Max |roll|: {jnp.max(jnp.abs(states_history[:, idx_roll])):.4f} rad")
    print(f"  Max |pitch|: {jnp.max(jnp.abs(states_history[:, idx_pitch])):.4f} rad")
    print(f"  Max |u_pitch|: {jnp.max(jnp.abs(actions_history[:, 0])):.4f}")
    print(f"  Max |u_roll|: {jnp.max(jnp.abs(actions_history[:, 1])):.4f}")
    
    # Create combined plot with all states, actions, and wheel angles
    num_states = state_dim
    num_actions = actions_history.shape[1]
    total_subplots = num_states + num_actions + 2  # +2 for drive and reaction wheel angles
    
    fig, axes = plt.subplots(total_subplots, 1, figsize=(12, 2.5 * total_subplots))
    if total_subplots == 1:
        axes = [axes]
    
    subplot_idx = 0
    
    # Plot all state variables
    for state_idx in range(num_states):
        ax = axes[subplot_idx]
        state_label = model_states_labels[state_idx] if model_states_labels is not None else f'State {state_idx}'
        
        ax.plot(time_steps, states_history[:, state_idx], 'b-', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)
        ax.set_ylabel(state_label)
        ax.set_title(f'{state_label} - Feedback Control ({training_type})')
        ax.grid(True, alpha=0.3)
        subplot_idx += 1
    
    # Plot all actions
    action_labels = model_actions_labels if model_actions_labels is not None else [f'Action {i}' for i in range(num_actions)]
    colors = ['b', 'r', 'g', 'm', 'c', 'y']
    
    for action_idx in range(num_actions):
        ax = axes[subplot_idx]
        color = colors[action_idx % len(colors)]
        
        ax.plot(time_steps, actions_history[:, action_idx], color=color, linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)
        ax.set_ylabel(action_labels[action_idx])
        ax.set_title(f'{action_labels[action_idx]} ({training_type})')
        ax.grid(True, alpha=0.3)
        subplot_idx += 1
    
    # Plot drive wheel angle
    ax = axes[subplot_idx]
    ax.plot(time_steps, drive_wheel_angles, 'g-', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)
    ax.set_ylabel('Drive Wheel Angle [rad]')
    ax.set_title(f'Drive Wheel Angle ({training_type})')
    ax.grid(True, alpha=0.3)
    subplot_idx += 1
    
    # Plot reaction wheel angle
    ax = axes[subplot_idx]
    ax.plot(time_steps, reaction_wheel_angles, 'm-', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)
    ax.set_ylabel('Reaction Wheel Angle [rad]')
    ax.set_title(f'Reaction Wheel Angle ({training_type})')
    ax.set_xlabel('Time [s]')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = f'plots/{save_prefix}_{training_type}_complete.pdf'
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Complete feedback control plot saved to {output_file}")
    
    return {
        "states": states_history,
        "actions": actions_history,
        "drive_wheel_angles": drive_wheel_angles,
        "reaction_wheel_angles": reaction_wheel_angles,
        "time": time_steps
    }


if __name__ == "__main__":
    dataset_path = "dataset/dataset_100_step.pkl"
    rollout_length = 50  # Matching the USER's recent manual change
    model_path = "models/trained_model_multistep.eqx"
    
    print("\n" + "=" * 70)
    print("Generating Paper Plots")
    print("=" * 70)
    
    plot_paper_comparison(
        model_path=model_path,
        dataset_path=dataset_path,
        rollout_length=rollout_length
    )
    
    print("\n" + "=" * 70)
    print("MULTI-STEP TRAINED MODEL (with curriculum)")
    print("=" * 70)
    mse_multistep = evaluate_rollout(
        model_path=model_path,
        dataset_path=dataset_path,
        rollout_length=rollout_length,
        save_prefix="rollout_50step"
    )
    
    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)
    print("=" * 70)
    mse_multistep = evaluate_rollout(
        model_path="models/trained_model_multistep.eqx",
        dataset_path=dataset_path,
        rollout_length=rollout_length,
        save_prefix="rollout_50step"
    )
    
    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)