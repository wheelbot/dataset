"""
Multi-step rollout training for dynamics models.

Trains dynamics models using a curriculum-based multi-step rollout loss.
The curriculum linearly increases rollout length from 1 to max_length during
the first half of training, then trains on full length for the second half.
"""

import os
import jax.numpy as jnp
import jax
import equinox as eqx
import optax as opx
import pickle
import matplotlib.pyplot as plt
import tqdm
from modellearning_common import (
    DynamicsDataset, create_mlp_model, save_dynamics_model,
    compute_normalization_params
)


def rollout_model(model, initial_state, actions_seq):
    """
    Roll out the dynamics model for multiple steps.
    
    Args:
        model: Dynamics model that predicts delta_state given [state, action]
        initial_state: Starting state (nx,) - should be normalized
        actions_seq: Sequence of actions (N, nu) - should be normalized
    
    Returns:
        predicted_states: (N, nx) predicted states at each timestep
    """
    def step_fn(state, action):
        model_input = jnp.concatenate([state, action])
        delta_state = model(model_input)
        next_state = state + delta_state
        return next_state, next_state
    
    _, predicted_states = jax.lax.scan(step_fn, initial_state, actions_seq)
    return predicted_states


def multistep_loss_fn(model, initial_states, actions_seqs, target_states, rollout_length):
    """
    Compute loss over multi-step rollouts with curriculum learning.
    
    Always rolls out for the full sequence length but masks out the loss
    for timesteps beyond rollout_length. This avoids JIT recompilation.
    
    Args:
        model: Dynamics model
        initial_states: (B, nx) batch of initial states
        actions_seqs: (B, T, nu) batch of action sequences
        target_states: (B, T, nx) batch of target state trajectories (as deltas from initial)
        rollout_length: Number of steps to include in loss (curriculum parameter)
    
    Returns:
        Mean squared error over active steps and batch
    """
    max_length = actions_seqs.shape[1]
    
    rollout_batch = jax.vmap(rollout_model, in_axes=(None, 0, 0))
    predicted_trajectories = rollout_batch(model, initial_states, actions_seqs)
    
    # Create mask: 1.0 for timesteps < rollout_length, 0.0 otherwise
    timesteps = jnp.arange(max_length)
    mask = (timesteps < rollout_length).astype(jnp.float32)
    mask = mask[None, :, None]  # Shape: (1, T, 1) for broadcasting
    
    # Compute squared errors and apply mask
    squared_errors = (predicted_trajectories - target_states) ** 2
    masked_errors = squared_errors * mask
    
    # Mean over active timesteps only
    num_active = rollout_length * initial_states.shape[0] * target_states.shape[2]
    return jnp.sum(masked_errors) / num_active


def get_curriculum_rollout_length(epoch, num_epochs, min_rollout_length, max_rollout_length):
    """
    Compute curriculum rollout length for current epoch.
    
    Linear schedule: increases from min_rollout_length to max_rollout_length in first half,
    then stays at max_rollout_length for second half.
    
    Args:
        epoch: Current epoch (0-indexed)
        num_epochs: Total number of epochs
        min_rollout_length: Starting rollout length
        max_rollout_length: Maximum rollout length to reach
        
    Returns:
        Current rollout length (integer)
    """
    halfway = num_epochs // 2
    
    if epoch < halfway:
        progress = epoch / halfway
        rollout_length = min_rollout_length + progress * (max_rollout_length - min_rollout_length)
    else:
        rollout_length = max_rollout_length
    
    return int(jnp.ceil(rollout_length))


def train_multistep(
    train_dataset,
    test_dataset,
    state_dim,
    action_dim,
    min_rollout_length=1,
    max_rollout_length=20,
    num_epochs=100,
    batch_size=1024,
    learning_rate=1e-3,
    mlp_width_size=64,
    mlp_depth=2,
    seed=0
):
    """
    Train dynamics model with curriculum-based multi-step rollout loss.
    
    The curriculum linearly increases the rollout length from min_rollout_length to 
    max_rollout_length during the first half of training epochs, then trains on the 
    full length for the second half.
    
    Args:
        train_dataset: Training DynamicsDataset
        test_dataset: Test DynamicsDataset
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        min_rollout_length: Starting rollout length for curriculum
        max_rollout_length: Maximum rollout length for training
        num_epochs: Number of training epochs
        batch_size: Batch size (should be smaller than one-step due to memory)
        learning_rate: Initial learning rate
        mlp_width_size: Width of MLP hidden layers
        mlp_depth: Number of MLP hidden layers
        seed: Random seed
        
    Returns:
        Tuple of (model, train_losses, test_losses, epoch_numbers, curriculum_schedule)
    """
    
    key = jax.random.PRNGKey(seed)
    
    # Ensure batch_size is not larger than dataset
    if batch_size > len(train_dataset):
        print(f"Warning: Batch size {batch_size} > dataset size {len(train_dataset)}. Clamping to dataset size.")
        batch_size = len(train_dataset)
    
    key, model_key = jax.random.split(key)
    model = create_mlp_model(state_dim, action_dim, mlp_width_size, mlp_depth, model_key)
    
    num_batches = max(1, len(train_dataset) // batch_size)
    num_steps = num_epochs * num_batches
    optimizer = opx.adamw(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    @eqx.filter_jit
    def train_step(model, opt_state, states, actions, next_states, rollout_length):
        def loss_fn(m):
            return multistep_loss_fn(m, states, actions, next_states, rollout_length)
        
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss
    
    def evaluate(model, dataset, rollout_length):
        return multistep_loss_fn(
            model, dataset.states, dataset.actions, dataset.next_states, rollout_length
        )
    
    train_losses = []
    test_losses = []
    epoch_numbers = []
    curriculum_schedule = []
    # num_batches is already calculated above
    
    print(f"Training with curriculum-based multi-step rollout")
    print(f"Rollout length: {min_rollout_length} → {max_rollout_length}, Batch size: {batch_size}")
    print(f"Curriculum: {min_rollout_length} → {max_rollout_length} (epochs 0-{num_epochs//2}), "
          f"then {max_rollout_length} (epochs {num_epochs//2+1}-{num_epochs})")
    print(f"Batches per epoch: {num_batches}")
    print()
    
    
    for epoch in range(num_epochs):
        current_rollout_length = get_curriculum_rollout_length(
            epoch, num_epochs, min_rollout_length, max_rollout_length
        )
        curriculum_schedule.append(current_rollout_length)
        
        epoch_loss = 0.0
        
        for batch_idx in tqdm.tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}"):
            key, batch_key = jax.random.split(key)
            batch_states, batch_actions, batch_next_states = train_dataset.get_batch(
                batch_size, batch_key
            )
            model, opt_state, loss = train_step(
                model, opt_state, batch_states, batch_actions, 
                batch_next_states, current_rollout_length
            )
            epoch_loss += loss
        
        epoch_loss /= num_batches
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            train_loss = evaluate(model, train_dataset, current_rollout_length)
            test_loss = evaluate(model, test_dataset, current_rollout_length)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            epoch_numbers.append(epoch + 1)
            print(f"Epoch {epoch+1:3d}/{num_epochs} [K={current_rollout_length:2d}] - "
                  f"Train: {train_loss:.6f}, Test: {test_loss:.6f}")
    
    final_train_loss = evaluate(model, train_dataset, max_rollout_length)
    final_test_loss = evaluate(model, test_dataset, max_rollout_length)
    print(f"\nFinal (K={max_rollout_length}) - Train: {final_train_loss:.6f}, "
          f"Test: {final_test_loss:.6f}")
    
    return model, train_losses, test_losses, epoch_numbers, curriculum_schedule


if __name__ == "__main__":
    with open("dataset/dataset_100_step.pkl", "rb") as f:
        data = pickle.load(f)
    
    train_data = data["train"]
    test_data = data["test"]
    
    MAX_ROLLOUT_LENGTH = 50
    MODEL_WIDTH_SIZE = 128
    MODEL_DEPTH = 3

    # shape of train data
    print(f"Train data shape: {train_data['states'].shape}")
    print(f"Train data shape: {train_data['nextstates'].shape}")
    
    train_states = jnp.squeeze(jnp.array(train_data["states"]))
    train_actions = jnp.array(train_data["actions"])[:, :MAX_ROLLOUT_LENGTH, :]
    train_next_states = jnp.array(train_data["nextstates"])[:, :MAX_ROLLOUT_LENGTH, :]
    
    test_states = jnp.squeeze(jnp.array(test_data["states"]))
    test_actions = jnp.array(test_data["actions"])[:, :MAX_ROLLOUT_LENGTH, :]
    test_next_states = jnp.array(test_data["nextstates"])[:, :MAX_ROLLOUT_LENGTH, :]

    states_labels = data.get("states_labels", [f"state_{i}" for i in range(train_states.shape[1])])
    actions_labels = data.get("actions_labels", [f"action_{i}" for i in range(train_actions.shape[2])])
    
    print(f"Dataset loaded: {train_states.shape[0]} train trajectories, {test_states.shape[0]} test trajectories")
    print(f"State dim: {train_states.shape[1]}, Action dim: {train_actions.shape[2]}")
    print(f"Trajectory length: {train_actions.shape[1]}")
    print(f"State fields: {states_labels}")
    print(f"Action fields: {actions_labels}")
    
    # Compute normalization from training data only
    state_mean, state_std, action_mean, action_std = compute_normalization_params(
        train_states, train_actions
    )
    
    def prepare_dataset(states, actions, next_states, state_mean, state_std, action_mean, action_std):
        states_norm = (states - state_mean) / state_std
        next_states_norm = (next_states - state_mean) / state_std
        actions_norm = (actions - action_mean) / action_std
        return DynamicsDataset(states_norm, actions_norm, next_states_norm, trajectory_length=MAX_ROLLOUT_LENGTH)

    train_dataset = prepare_dataset(train_states, train_actions, train_next_states, state_mean, state_std, action_mean, action_std)
    test_dataset = prepare_dataset(test_states, test_actions, test_next_states, state_mean, state_std, action_mean, action_std)
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print()
    
    state_dim = train_states.shape[1]
    action_dim = train_actions.shape[2]
    
    model, train_losses, test_losses, epoch_numbers, curriculum_schedule = train_multistep(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        state_dim=state_dim,
        action_dim=action_dim,
        min_rollout_length=5,
        max_rollout_length=MAX_ROLLOUT_LENGTH,
        num_epochs=400,
        batch_size=1024,
        learning_rate=1e-4,
        mlp_width_size=MODEL_WIDTH_SIZE,
        mlp_depth=MODEL_DEPTH,
        seed=42
    )
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(epoch_numbers, train_losses, label='Train Loss', marker='o')
    ax1.plot(epoch_numbers, test_losses, label='Test Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Multi-step Training Loss with Curriculum')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(range(1, len(curriculum_schedule) + 1), curriculum_schedule, 
             marker='o', markersize=3, linewidth=2, color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Rollout Length')
    ax2.set_title('Curriculum Schedule')
    ax2.grid(True)
    ax2.set_ylim([0, MAX_ROLLOUT_LENGTH + 1])

    # Create output directory if it doesn't exist
    os.makedirs("plots/", exist_ok=True)
    
    plt.tight_layout()
    plt.savefig('plots/loss_multistep_curriculum.pdf')
    plt.close()
    print("\nLoss plot saved to plots/loss_multistep_curriculum.pdf")
    
    hyperparams = {
        "input_dim": state_dim + action_dim,
        "output_dim": state_dim,
        "width_size": MODEL_WIDTH_SIZE,
        "depth": MODEL_DEPTH,
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "states_labels": states_labels,
        "actions_labels": actions_labels,
        "training_type": "multistep_curriculum",
        "max_rollout_length": MAX_ROLLOUT_LENGTH,
        "num_epochs": 100,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "seed": 0
    }
    
    save_dynamics_model("models/trained_model_multistep.eqx", model, hyperparams)
    print("Multi-step trained model saved to models/trained_model_multistep.eqx")
    
    