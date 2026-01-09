"""
Physics parameter learning for the wheelbot EOM using multi-step rollout.

Trains physics parameters using curriculum-based multi-step rollout loss.
This module provides generic training functions that work with any physics model.
"""

import jax.numpy as jnp
import jax
import equinox as eqx
import optax as opx
import tqdm


def multistep_physics_loss_fn(physics_params, initial_states_sim, actions_seqs, 
                               target_states_dataset, state_labels, dt, rollout_length,
                               rollout_fn, state_to_obs_fn, state_weights=None):
    """
    Compute loss over multi-step physics rollouts with curriculum learning.
    
    Args:
        physics_params: PhysicsParameters object
        initial_states_sim: (B, state_dim) batch of initial states in simulator format
        actions_seqs: (B, T, action_dim) batch of action sequences
        target_states_dataset: (B, T, n_obs) batch of target trajectories in dataset format
        state_labels: List of state labels for conversion
        dt: Time step
        rollout_length: Number of steps to include in loss (curriculum parameter)
        rollout_fn: Function (params, initial_state, actions, dt) -> states trajectory
        state_to_obs_fn: Function (state_vector, state_labels) -> observation vector
        state_weights: Optional (n_obs,) array of weights for each state dimension.
                       If None, uniform weights are used.
    
    Returns:
        Mean squared error over active steps and batch
    """
    max_length = actions_seqs.shape[1]
    n_obs = target_states_dataset.shape[2]
    
    # Vectorize rollout over batch
    rollout_batch = jax.vmap(rollout_fn, in_axes=(None, 0, 0, None))
    predicted_trajectories_sim = rollout_batch(physics_params, initial_states_sim, actions_seqs, dt)
    
    # Convert predicted trajectories from simulator format to dataset format
    def convert_traj(traj_sim):
        return jax.vmap(state_to_obs_fn, in_axes=(0, None))(traj_sim, state_labels)
    
    predicted_trajectories_dataset = jax.vmap(convert_traj)(predicted_trajectories_sim)
    
    # Create temporal mask: 1.0 for timesteps < rollout_length, 0.0 otherwise
    timesteps = jnp.arange(max_length)
    mask = (timesteps < rollout_length).astype(jnp.float32)
    mask = mask[None, :, None]  # Shape: (1, T, 1) for broadcasting
    
    # Apply state weights if provided
    if state_weights is None:
        state_weights = jnp.ones(n_obs)
    state_weights = jnp.reshape(state_weights, (1, 1, n_obs))  # Shape: (1, 1, n_obs)
    
    # Compute weighted squared errors and apply mask
    squared_errors = (predicted_trajectories_dataset - target_states_dataset) ** 2
    weighted_errors = squared_errors * state_weights
    masked_errors = weighted_errors * mask
    
    # Mean over active timesteps only (normalized by sum of weights)
    num_active = rollout_length * initial_states_sim.shape[0]
    weight_sum = jnp.sum(state_weights)
    return jnp.sum(masked_errors) / (num_active * weight_sum)


def get_curriculum_rollout_length(epoch, num_epochs, min_rollout_length, max_rollout_length):
    """
    Compute curriculum rollout length for current epoch.
    
    Three-stage schedule:
    - Epochs 0-9: min_rollout_length
    - Epochs 10-59: (max_rollout_length - min_rollout_length) / 2 + min_rollout_length
    - Epochs 60+: max_rollout_length
    """
    if epoch < 10:
        rollout_length = min_rollout_length
    elif epoch < 60:
        rollout_length = (max_rollout_length - min_rollout_length) / 2 + min_rollout_length
    else:
        rollout_length = max_rollout_length
    
    return int(rollout_length)


def train_physics_params(
    physics_params_init,
    states_initial,
    actions_trajectories,
    states_targets,
    state_labels,
    dt,
    rollout_fn,
    state_to_obs_fn,
    state_dict_to_vector_fn,
    state_weights=None,
    min_rollout_length=1,
    max_rollout_length=20,
    num_epochs=100,
    batch_size=1000,
    learning_rate=1e-3,
    seed=0
):
    """
    Train physics parameters with curriculum-based multi-step rollout loss.
    
    Args:
        physics_params_init: Initial PhysicsParameters object
        states_initial: Initial states (N, n_obs) in dataset format
        actions_trajectories: Action sequences (N, T, action_dim)
        states_targets: Target state trajectories (N, T, n_obs) in dataset format
        state_labels: List of state label names
        dt: Time step from dataset
        rollout_fn: Function (params, initial_state, actions, dt) -> states trajectory
        state_to_obs_fn: Function (state_vector, state_labels) -> observation vector
        state_dict_to_vector_fn: Function (state_dict, state_labels) -> state vector
        state_weights: Optional (n_obs,) array of weights for each state dimension.
                       If None, uniform weights are used.
        min_rollout_length: Starting rollout length for curriculum
        max_rollout_length: Maximum rollout length for training
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        seed: Random seed
        
    Returns:
        Tuple of (physics_params, train_losses, epoch_numbers, curriculum_schedule)
    """
    
    key = jax.random.PRNGKey(seed)
    
    physics_params = physics_params_init
    
    # Convert initial states to simulator format
    def convert_initial_state(state_obs):
        state_dict = {label: state_obs[i] for i, label in enumerate(state_labels)}
        return state_dict_to_vector_fn(state_dict, state_labels)
    
    states_initial_sim = jax.vmap(convert_initial_state)(states_initial)
    
    # Setup optimizer
    batch_size = min(len(states_initial), batch_size)
    num_batches = len(states_initial) // batch_size
    num_steps = num_epochs * num_batches
    
    # Learning rate schedule: constant during curriculum ramp, then cosine decay
    ramp_epochs = max_rollout_length - min_rollout_length
    ramp_steps = ramp_epochs * num_batches
    decay_steps = num_steps - ramp_steps
    
    # schedule = opx.join_schedules(
    #     schedules=[
    #         opx.constant_schedule(learning_rate),
    #         opx.cosine_decay_schedule(learning_rate, max(decay_steps, 1))
    #     ],
    #     boundaries=[ramp_steps]
    # )
    optimizer = opx.adamw(learning_rate=learning_rate)
    opt_state = optimizer.init(eqx.filter(physics_params, eqx.is_array))
    
    @eqx.filter_jit
    def train_step(params, opt_state, states_init, actions, targets, rollout_length):
        def loss_fn(p):
            return multistep_physics_loss_fn(
                p, states_init, actions, targets, state_labels, dt, rollout_length,
                rollout_fn, state_to_obs_fn, state_weights
            )
        
        loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return params, new_opt_state, loss
    
    train_losses = []
    epoch_numbers = []
    curriculum_schedule = []
    
    print(f"Training physics parameters with curriculum-based multi-step rollout")
    print(f"Rollout length: {min_rollout_length} → {max_rollout_length}")
    print(f"Number of trajectories: {len(states_initial)}")
    print(f"Trajectory length: {actions_trajectories.shape[1]}")
    print(f"Batch size: {batch_size}, Batches per epoch: {num_batches}")
    print(f"Time step dt: {dt}")
    print()
    
    for epoch in range(num_epochs):
        current_rollout_length = get_curriculum_rollout_length(
            epoch, num_epochs, min_rollout_length, max_rollout_length
        )
        curriculum_schedule.append(current_rollout_length)
        
        # Shuffle data at the beginning of each epoch
        key, shuffle_key = jax.random.split(key)
        indices = jax.random.permutation(shuffle_key, len(states_initial))
        
        epoch_loss = 0.0
        
        # Train on batches
        for batch_idx in tqdm.tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_start = batch_idx * batch_size
            batch_end = batch_start + batch_size
            batch_indices = indices[batch_start:batch_end]
            
            batch_states_init = states_initial_sim[batch_indices]
            batch_actions = actions_trajectories[batch_indices]
            batch_targets = states_targets[batch_indices]
            
            physics_params, opt_state, loss = train_step(
                physics_params, opt_state, batch_states_init,
                batch_actions, batch_targets, current_rollout_length
            )
            
            epoch_loss += loss
        
        epoch_loss /= num_batches
        train_losses.append(epoch_loss)
        epoch_numbers.append(epoch + 1)
        
        # Get current learning rate
        # current_step = epoch * num_batches
        # current_lr = schedule(current_step)
        current_lr = learning_rate
        
        # Print epoch summary with parameters (single line)
        params_str = physics_params.params_string() if hasattr(physics_params, 'params_string') else ""
        print(f"Epoch {epoch+1:3d}/{num_epochs} [K={current_rollout_length:2d}] Loss: {epoch_loss:.6f} LR: {current_lr:.2e} | {params_str}")
    
    print(f"\nFinal (K={max_rollout_length}) - Loss: {train_losses[-1]:.6f}")
    
    return physics_params, train_losses, epoch_numbers, curriculum_schedule
