"""
Common utilities for dynamics model learning.

Provides shared functionality for dataset handling, model creation,
normalization, and model serialization.
"""

import jax.numpy as jnp
import jax
import equinox as eqx
import json
import os


class DynamicsDataset:
    """
    Dataset for dynamics model training with support for trajectories.
    
    Supports both one-step prediction (trajectory_length=1) and multi-step
    rollout training (trajectory_length>1).
    """
    
    def __init__(self, states, actions, next_states, trajectory_length=1):
        """
        Initialize dynamics dataset.
        
        Args:
            states: Initial states (N, nx) for one-step or (N, nx) for multi-step
            actions: Actions (N, nu) for one-step or (N, T, nu) for multi-step
            next_states: Next states (N, nx) for one-step or (N, T, nx) for multi-step
            trajectory_length: Length of trajectories (1 for one-step prediction)
        """
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.n_samples = len(states)
        self.trajectory_length = trajectory_length
    
    def __len__(self):
        """Get number of samples in dataset."""
        return self.n_samples
    
    def get_batch(self, batch_size, key):
        """
        Sample a random batch from the dataset.
        
        Args:
            batch_size: Number of samples to retrieve
            key: JAX random key for sampling
            
        Returns:
            Tuple of (states, actions, next_states) for the batch
        """
        indices = jax.random.choice(key, self.n_samples, (batch_size,), replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.next_states[indices]
        )


def create_mlp_model(state_dim, action_dim, width_size=64, depth=2, key=None):
    """
    Create a standard MLP dynamics model.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        width_size: Width of hidden layers
        depth: Number of hidden layers
        key: JAX random key for initialization
        
    Returns:
        Equinox MLP model that predicts state deltas
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    input_dim = state_dim + action_dim
    output_dim = state_dim
    
    return eqx.nn.MLP(
        in_size=input_dim,
        out_size=output_dim,
        width_size=width_size,
        depth=depth,
        key=key
    )


def save_dynamics_model(filename: str, model: eqx.Module, hyperparams: dict):
    """
    Save model weights and hyperparameters to file.
    
    Args:
        filename: Path to save the model
        model: Equinox model to save
        hyperparams: Dictionary of hyperparameters to save alongside model
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        json_str = json.dumps(hyperparams)
        f.write((json_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load_dynamics_model(filename: str):
    """
    Load model and hyperparameters from file.
    
    Args:
        filename: Path to the saved model file
        
    Returns:
        Tuple of (model, hyperparams)
    """
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        
        key = jax.random.PRNGKey(hyperparams.pop("seed", 0))
        model = eqx.nn.MLP(
            in_size=hyperparams["input_dim"],
            out_size=hyperparams["output_dim"],
            width_size=hyperparams["width_size"],
            depth=hyperparams["depth"],
            key=key
        )
        
        model = eqx.tree_deserialise_leaves(f, model)
        return model, hyperparams


def compute_normalization_params(states, actions):
    """
    Compute normalization parameters from data.
    
    Args:
        states: State data (N, nx) or (N, T, nx)
        actions: Action data (N, nu) or (N, T, nu)
        
    Returns:
        Tuple of (state_mean, state_std, action_mean, action_std)
    """
    if states.ndim == 3:
        states = states.reshape(-1, states.shape[-1])
    if actions.ndim == 3:
        actions = actions.reshape(-1, actions.shape[-1])
    
    state_mean = jnp.mean(states, axis=0)
    state_std = jnp.std(states, axis=0) + 1e-8
    action_mean = jnp.mean(actions, axis=0)
    action_std = jnp.std(actions, axis=0) + 1e-8
    
    return state_mean, state_std, action_mean, action_std


def normalize_data(states, actions, next_states, state_mean, state_std, action_mean, action_std):
    """
    Normalize states and actions using provided parameters.
    
    Args:
        states: State data to normalize
        actions: Action data to normalize
        next_states: Next state data to normalize
        state_mean: Mean for state normalization
        state_std: Standard deviation for state normalization
        action_mean: Mean for action normalization
        action_std: Standard deviation for action normalization
        
    Returns:
        Tuple of (states_norm, actions_norm, next_states_norm)
    """
    states_norm = (states - state_mean) / state_std
    next_states_norm = (next_states - state_mean) / state_std
    actions_norm = (actions - action_mean) / action_std
    
    return states_norm, actions_norm, next_states_norm


