"""
One-step prediction training for dynamics models.

Trains dynamics models using standard supervised learning on one-step predictions.
"""

import jax.numpy as jnp
import jax
import equinox as eqx
import optax as opx
import pickle
import matplotlib.pyplot as plt
from modellearning_common import (
    DynamicsDataset, create_mlp_model, save_dynamics_model, 
    load_dynamics_model, compute_normalization_params
)

if __name__=="__main__":
    with open("dataset/dataset_1_step.pkl", "rb") as f:
        data = pickle.load(f)
    
    states = jnp.squeeze(jnp.array(data["states"]))
    actions = jnp.squeeze(jnp.array(data["actions"]))
    next_states = jnp.squeeze(jnp.array(data["nextstates"]))
    states_labels = data.get("states_labels", [f"state_{i}" for i in range(states.shape[1])])
    actions_labels = data.get("actions_labels", [f"action_{i}" for i in range(actions.shape[1])])
    
    print(f"Dataset loaded: {states.shape[0]} samples")
    print(f"State dim: {states.shape[1]}, Action dim: {actions.shape[1]}")
    print(f"State fields: {states_labels}")
    print(f"Action fields: {actions_labels}")
    
    state_mean, state_std, action_mean, action_std = compute_normalization_params(
        states, actions
    )

    states = (states - state_mean) / state_std
    next_states = (next_states - state_mean) / state_std
    actions = (actions - action_mean) / action_std
    
    n_samples = len(states)
    key = jax.random.PRNGKey(0)
    indices = jax.random.permutation(key, n_samples)
    
    states = states[indices]
    actions = actions[indices]
    next_states = next_states[indices]
    
    split_idx = int(0.8 * n_samples)
    train_states = states[:split_idx]
    train_actions = actions[:split_idx]
    train_next_states = next_states[:split_idx]
    
    test_states = states[split_idx:]
    test_actions = actions[split_idx:]
    test_next_states = next_states[split_idx:]
    
    train_dataset = DynamicsDataset(train_states, train_actions, train_next_states)
    test_dataset = DynamicsDataset(test_states, test_actions, test_next_states)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print()
    
    state_dim = train_states.shape[1]
    action_dim = train_actions.shape[1]

    mlp_width_size = 64
    mlp_depth = 2
    key, model_key = jax.random.split(key)
    model = create_mlp_model(state_dim, action_dim, mlp_width_size, mlp_depth, model_key)

    num_epochs = 100
    batch_size = 1024
    learning_rate = 1e-3
    num_steps = num_epochs * (len(train_dataset) // batch_size)
    print(f"Batches per epoch: {len(train_dataset) // batch_size}")
    print(f"Training one-step prediction model")
    print()

    schedule = opx.cosine_decay_schedule(learning_rate, num_steps)
    optimizer = opx.adamw(schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def loss_fn(model, states, actions, next_states):
        model_input = jnp.concatenate([states, actions], axis=-1)
        delta_state = jax.vmap(model)(model_input)
        predicted_states = states + delta_state
        return jnp.mean((predicted_states - next_states) ** 2)

    @eqx.filter_jit
    def train_step(model, opt_state, states, actions, next_states):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, states, actions, next_states)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    def evaluate(model, dataset):
        model_input = jnp.concatenate([dataset.states, dataset.actions], axis=-1)
        delta_state = jax.vmap(model)(model_input)
        predicted_states = dataset.states + delta_state
        return jnp.mean((predicted_states - dataset.next_states) ** 2)

    train_losses = []
    test_losses = []
    epoch_numbers = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = len(train_dataset) // batch_size
        
        for batch_idx in range(num_batches):
            key, batch_key = jax.random.split(key)
            batch_states, batch_actions, batch_next_states = train_dataset.get_batch(batch_size, batch_key)
            model, opt_state, loss = train_step(model, opt_state, batch_states, batch_actions, batch_next_states)
            epoch_loss += loss
        
        epoch_loss /= num_batches
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            train_loss = evaluate(model, train_dataset)
            test_loss = evaluate(model, test_dataset)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            epoch_numbers.append(epoch + 1)
            print(f"Epoch {epoch+1:3d}/{num_epochs} - Train: {train_loss:.6f}, Test: {test_loss:.6f}")

    train_loss = evaluate(model, train_dataset)
    test_loss = evaluate(model, test_dataset)
    print(f"\nFinal - Train: {train_loss:.6f}, Test: {test_loss:.6f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_numbers, train_losses, label='Train Loss', marker='o')
    plt.plot(epoch_numbers, test_losses, label='Test Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('One-step Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/loss_onestep.pdf')
    plt.close()
    print("\nLoss plot saved to plots/loss_onestep.pdf")
    
    hyperparams = {
        "input_dim": state_dim + action_dim,
        "output_dim": state_dim,
        "width_size": mlp_width_size,
        "depth": mlp_depth,
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "states_labels": states_labels,
        "actions_labels": actions_labels,
        "training_type": "onestep",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "seed": 0
    }
    
    save_dynamics_model("models/trained_model_onestep.eqx", model, hyperparams)
    print("Model saved to models/trained_model_onestep.eqx")
    
    loaded_model, loaded_hyperparams = load_dynamics_model("models/trained_model_onestep.eqx")
    print("Model verified: loaded successfully")
    
    
    