"""
Transformer-based timeseries classification for surface detection.

Trains a tiny transformer model to classify surfaces (black_pvc, concrete, gray_felt)
based on IMU sensor data (gyroscope and accelerometer readings).
"""

import jax.numpy as jnp
import jax
import equinox as eqx
import optax as opx
import pickle
import matplotlib.pyplot as plt
import tqdm
import os


class PositionalEncoding(eqx.Module):
    """Learnable positional encoding for transformer."""
    
    encoding: jnp.ndarray
    
    def __init__(self, max_len, d_model, key):
        """
        Initialize positional encoding.
        
        Args:
            max_len: Maximum sequence length
            d_model: Model dimension
            key: JAX random key
        """
        # Learnable positional encoding
        self.encoding = jax.random.normal(key, (max_len, d_model)) * 0.02
    
    def __call__(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (seq_len, d_model)
            
        Returns:
            x with positional encoding added
        """
        seq_len = x.shape[0]
        return x + self.encoding[:seq_len]


class TransformerBlock(eqx.Module):
    """Single transformer encoder block."""
    
    attention: eqx.nn.MultiheadAttention
    norm1: eqx.nn.LayerNorm
    ffn: eqx.nn.MLP
    norm2: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout
    
    def __init__(self, d_model, num_heads, mlp_dim, dropout_rate, key):
        """
        Initialize transformer block.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            mlp_dim: Dimension of feedforward network
            dropout_rate: Dropout rate
            key: JAX random key
        """
        key1, key2, key3 = jax.random.split(key, 3)
        
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=d_model,
            key=key1
        )
        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.ffn = eqx.nn.MLP(
            in_size=d_model,
            out_size=d_model,
            width_size=mlp_dim,
            depth=1,
            activation=jax.nn.gelu,
            key=key2
        )
        self.norm2 = eqx.nn.LayerNorm(d_model)
        self.dropout = eqx.nn.Dropout(dropout_rate)
    
    def __call__(self, x, *, key=None):
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (seq_len, d_model)
            key: JAX random key for dropout
            
        Returns:
            Output tensor of same shape as input
        """
        # Multi-head self-attention with residual connection
        key1, key2, key3 = jax.random.split(key, 3) if key is not None else (None, None, None)
        
        attn_out = self.attention(x, x, x, key=key1)
        if key is not None:
            attn_out = self.dropout(attn_out, key=key2)
        x = jax.vmap(self.norm1)(x + attn_out)
        
        # Feedforward network with residual connection
        ffn_out = jax.vmap(self.ffn)(x)
        if key is not None:
            ffn_out = self.dropout(ffn_out, key=key3)
        x = jax.vmap(self.norm2)(x + ffn_out)
        
        return x


class TimeSeriesTransformer(eqx.Module):
    """Transformer model for timeseries classification."""
    
    input_projection: eqx.nn.Linear
    pos_encoding: PositionalEncoding
    transformer_blocks: list
    classifier: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    
    def __init__(
        self, 
        input_dim, 
        d_model, 
        num_heads, 
        num_layers, 
        mlp_dim, 
        num_classes,
        max_seq_len,
        dropout_rate=0.1,
        key=None
    ):
        """
        Initialize transformer classifier.
        
        Args:
            input_dim: Dimension of input features
            d_model: Model dimension (embedding size)
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            mlp_dim: Dimension of feedforward network in transformer
            num_classes: Number of output classes
            max_seq_len: Maximum sequence length
            dropout_rate: Dropout rate
            key: JAX random key
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        
        keys = jax.random.split(key, num_layers + 3)
        
        # Project input features to model dimension
        self.input_projection = eqx.nn.Linear(input_dim, d_model, key=keys[0])
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(max_seq_len, d_model, keys[1])
        
        # Stack of transformer blocks
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, mlp_dim, dropout_rate, keys[2 + i])
            for i in range(num_layers)
        ]
        
        # Classification head
        self.classifier = eqx.nn.Linear(d_model, num_classes, key=keys[-1])
        self.dropout = eqx.nn.Dropout(dropout_rate)
    
    def __call__(self, x, seq_len=None, *, key=None):
        """
        Forward pass through transformer.
        
        Args:
            x: Input tensor of shape (seq_len, input_dim)
            seq_len: Actual sequence length to use (for masking if < max_seq_len)
            key: JAX random key for dropout
            
        Returns:
            Class logits of shape (num_classes,)
        """
        if seq_len is None:
            seq_len = x.shape[0]
        
        # Only use the first seq_len timesteps
        x = x[:seq_len]
        
        # Project input to model dimension
        x = jax.vmap(self.input_projection)(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply dropout to embeddings
        if key is not None:
            key, dropout_key = jax.random.split(key)
            x = self.dropout(x, key=dropout_key)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            if key is not None:
                key, block_key = jax.random.split(key)
            else:
                block_key = None
            x = block(x, key=block_key)
        
        # Global average pooling over sequence dimension
        x = jnp.mean(x, axis=0)
        
        # Classification head
        logits = self.classifier(x)
        
        return logits


class ClassificationDataset:
    """Dataset for timeseries classification."""
    
    def __init__(self, timeseries, labels):
        """
        Initialize classification dataset.
        
        Args:
            timeseries: Time series data of shape (N, seq_len, input_dim)
            labels: Class labels of shape (N,)
        """
        self.timeseries = timeseries
        self.labels = labels
        self.n_samples = len(labels)
    
    def __len__(self):
        return self.n_samples
    
    def get_batch(self, batch_size, key):
        """
        Sample a random batch from the dataset.
        
        Args:
            batch_size: Number of samples to retrieve
            key: JAX random key for sampling
            
        Returns:
            Tuple of (timeseries, labels) for the batch
        """
        indices = jax.random.choice(key, self.n_samples, (batch_size,), replace=False)
        return self.timeseries[indices], self.labels[indices]


def compute_normalization_params(timeseries):
    """
    Compute mean and std for normalization across all samples and timesteps.
    
    Args:
        timeseries: Array of shape (N, T, D)
        
    Returns:
        Tuple of (mean, std) where each has shape (D,)
    """
    # Compute statistics across both samples and time dimensions
    mean = jnp.mean(timeseries, axis=(0, 1))
    std = jnp.std(timeseries, axis=(0, 1))
    # Avoid division by zero
    std = jnp.where(std < 1e-8, 1.0, std)
    return mean, std


def cross_entropy_loss(logits, labels):
    """
    Compute cross-entropy loss.
    
    Args:
        logits: Model predictions of shape (batch_size, num_classes)
        labels: Ground truth labels of shape (batch_size,)
        
    Returns:
        Scalar loss value
    """
    # Convert labels to one-hot encoding
    num_classes = logits.shape[-1]
    one_hot_labels = jax.nn.one_hot(labels, num_classes)
    
    # Compute cross-entropy loss
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(one_hot_labels * log_probs) / logits.shape[0]
    
    return loss


def compute_accuracy(logits, labels):
    """
    Compute classification accuracy.
    
    Args:
        logits: Model predictions of shape (batch_size, num_classes)
        labels: Ground truth labels of shape (batch_size,)
        
    Returns:
        Accuracy as a scalar between 0 and 1
    """
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)
    return accuracy


def train_transformer(
    train_dataset,
    test_dataset,
    input_dim,
    num_classes,
    d_model=64,
    num_heads=4,
    num_layers=2,
    mlp_dim=128,
    max_seq_len=50,
    num_epochs=100,
    batch_size=128,
    learning_rate=1e-3,
    dropout_rate=0.1,
    seed=0
):
    """
    Train transformer model for timeseries classification.
    
    Args:
        train_dataset: Training ClassificationDataset
        test_dataset: Test ClassificationDataset
        input_dim: Dimension of input features
        num_classes: Number of output classes
        d_model: Model embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        mlp_dim: Dimension of feedforward network
        max_seq_len: Maximum sequence length to use
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        dropout_rate: Dropout rate
        seed: Random seed
        
    Returns:
        Tuple of (model, train_losses, test_losses, train_accs, test_accs, epoch_numbers)
    """
    key = jax.random.PRNGKey(seed)
    
    # Create model
    key, model_key = jax.random.split(key)
    model = TimeSeriesTransformer(
        input_dim=input_dim,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_dim=mlp_dim,
        num_classes=num_classes,
        max_seq_len=max_seq_len,
        dropout_rate=dropout_rate,
        key=model_key
    )
    
    # Setup optimizer with cosine decay
    num_steps = num_epochs * (len(train_dataset) // batch_size)
    # schedule = opx.cosine_decay_schedule(learning_rate, num_steps)
    optimizer = opx.adamw(learning_rate=learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    @eqx.filter_jit
    def train_step(model, opt_state, x_batch, y_batch, key):
        """Single training step."""
        def loss_fn(m):
            # Batch forward pass with dropout enabled
            keys = jax.random.split(key, x_batch.shape[0])
            logits = jax.vmap(lambda x, k: m(x, seq_len=max_seq_len, key=k))(x_batch, keys)
            return cross_entropy_loss(logits, y_batch)
        
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss
    
    @eqx.filter_jit
    def evaluate(model, x_batch, y_batch):
        """Evaluate model on a batch (no dropout)."""
        # Batch forward pass without dropout
        logits = jax.vmap(lambda x: model(x, seq_len=max_seq_len, key=None))(x_batch)
        loss = cross_entropy_loss(logits, y_batch)
        acc = compute_accuracy(logits, y_batch)
        return loss, acc
    
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    epoch_numbers = []
    num_batches = len(train_dataset) // batch_size
    
    print(f"Training Transformer for Surface Classification")
    print(f"Model: d_model={d_model}, heads={num_heads}, layers={num_layers}")
    print(f"Sequence length: {max_seq_len}, Batch size: {batch_size}")
    print(f"Batches per epoch: {num_batches}")
    print()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx in tqdm.tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}"):
            key, batch_key, train_key = jax.random.split(key, 3)
            x_batch, y_batch = train_dataset.get_batch(batch_size, batch_key)
            model, opt_state, loss = train_step(model, opt_state, x_batch, y_batch, train_key)
            epoch_loss += loss
        
        epoch_loss /= num_batches
        
        # Evaluate every 10 epochs or on the first/last epoch
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            # Evaluate on full datasets (or large sample if too big)
            eval_size = min(1000, len(train_dataset))
            key, train_eval_key = jax.random.split(key)
            train_x, train_y = train_dataset.get_batch(eval_size, train_eval_key)
            train_loss, train_acc = evaluate(model, train_x, train_y)
            
            eval_size = min(1000, len(test_dataset))
            key, test_eval_key = jax.random.split(key)
            test_x, test_y = test_dataset.get_batch(eval_size, test_eval_key)
            test_loss, test_acc = evaluate(model, test_x, test_y)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            epoch_numbers.append(epoch + 1)
            
            print(f"Epoch {epoch+1:3d}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    return model, train_losses, test_losses, train_accs, test_accs, epoch_numbers


def save_model(filename, model, hyperparams):
    """Save model and hyperparameters."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    eqx.tree_serialise_leaves(filename, model)
    
    json_filename = filename.replace('.eqx', '_hyperparams.json')
    import json
    with open(json_filename, 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    print(f"Model saved to {filename}")
    print(f"Hyperparameters saved to {json_filename}")


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train transformer for timeseries classification")
    parser.add_argument(
        "--task",
        type=str,
        default="surface",
        choices=["surface", "robot", "group"],
        help="Classification task: 'surface' (surface type), 'robot' (robot ID), or 'group' (autonomous vs human)"
    )
    args = parser.parse_args()
    
    # Load dataset based on task
    task = args.task
    dataset_files = {
        "surface": "dataset/surface_classification_dataset.pkl",
        "robot": "dataset/robot_classification_dataset.pkl",
        "group": "dataset/group_classification_dataset.pkl"
    }
    
    label_keys = {
        "surface": "surfaces",
        "robot": "robots",
        "group": "categories"
    }
    
    print(f"=" * 80)
    print(f"TRAINING TRANSFORMER FOR {task.upper()} CLASSIFICATION")
    print(f"=" * 80)
    print()
    
    dataset_file = dataset_files[task]
    label_key = label_keys[task]
    
    with open(dataset_file, "rb") as f:
        data = pickle.load(f)
    
    train_timeseries = jnp.array(data["train_timeseries"])
    train_labels = jnp.array(data["train_labels"])
    test_timeseries = jnp.array(data["test_timeseries"])
    test_labels = jnp.array(data["test_labels"])
    
    print(f"Dataset loaded: {dataset_file}")
    print(f"  Train: {train_timeseries.shape[0]} samples")
    print(f"  Test: {test_timeseries.shape[0]} samples")
    print(f"  Sequence length: {train_timeseries.shape[1]}")
    print(f"  Input features: {train_timeseries.shape[2]}")
    print(f"  Classes: {data[label_key]}")
    print()
    
    # Compute normalization parameters from training data only
    mean, std = compute_normalization_params(train_timeseries)
    print(f"Normalization computed - Mean shape: {mean.shape}, Std shape: {std.shape}")
    
    # Normalize data
    train_timeseries_norm = (train_timeseries - mean) / std
    test_timeseries_norm = (test_timeseries - mean) / std
    
    # Create datasets
    train_dataset = ClassificationDataset(train_timeseries_norm, train_labels)
    test_dataset = ClassificationDataset(test_timeseries_norm, test_labels)
    
    # Training parameters
    D_MODEL = 64
    NUM_HEADS = 4
    NUM_LAYERS = 2
    MLP_DIM = 128
    BATCH_SIZE = 1024
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    DROPOUT_RATE = 0.1
    
    input_dim = train_timeseries.shape[2]
    num_classes = data["n_classes"]
    
    # Sequence lengths to evaluate
    SEQUENCE_LENGTHS = [2, 5, 10, 20, 50]
    
    # Store results for all sequence lengths
    all_results = {}
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    print("=" * 80)
    print("Training models for different sequence lengths")
    print("=" * 80)
    
    for seq_len in SEQUENCE_LENGTHS:
        print(f"\n{'=' * 80}")
        print(f"Training with sequence length: {seq_len}")
        print(f"{'=' * 80}\n")
        
        # Train model
        model, train_losses, test_losses, train_accs, test_accs, epoch_numbers = train_transformer(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            mlp_dim=MLP_DIM,
            max_seq_len=seq_len,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            dropout_rate=DROPOUT_RATE,
            seed=42
        )
        
        # Store results
        all_results[seq_len] = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "train_accs": train_accs,
            "test_accs": test_accs,
            "epoch_numbers": epoch_numbers,
            "final_train_acc": float(train_accs[-1]),
            "final_test_acc": float(test_accs[-1])
        }
        
        # Save model with sequence length in filename
        hyperparams = {
            "input_dim": int(input_dim),
            "num_classes": int(num_classes),
            "d_model": D_MODEL,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "mlp_dim": MLP_DIM,
            "max_seq_len": seq_len,
            "dropout_rate": DROPOUT_RATE,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "labels": data[label_key],
            "fields_states": data["fields_states"],
            "task": task
        }
        
        model_filename = f"models/{task}_classifier_seq{seq_len}.eqx"
        save_model(model_filename, model, hyperparams)
        
        # Plot individual training curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1.plot(epoch_numbers, train_losses, label='Train Loss', marker='o')
        ax1.plot(epoch_numbers, test_losses, label='Test Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (Cross-Entropy)')
        ax1.set_title(f'{task.capitalize()} Classification - Loss (seq_len={seq_len})')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epoch_numbers, train_accs, label='Train Accuracy', marker='o')
        ax2.plot(epoch_numbers, test_accs, label='Test Accuracy', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'{task.capitalize()} Classification - Accuracy (seq_len={seq_len})')
        ax2.legend()
        ax2.grid(True)
        ax2.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(f'plots/{task}_training_seq{seq_len}.pdf')
        plt.close()
        print(f"Training curves saved to plots/{task}_training_seq{seq_len}.pdf")
        
        print(f"\nResults for seq_len={seq_len}:")
        print(f"  Final Train Accuracy: {train_accs[-1]:.4f}")
        print(f"  Final Test Accuracy: {test_accs[-1]:.4f}")
    
    # Save all results to pickle for later analysis
    with open(f"results/{task}_sequence_length_comparison.pkl", "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nAll results saved to results/{task}_sequence_length_comparison.pkl")
    
    # Create comparison plot: Accuracy vs Sequence Length
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    seq_lengths = sorted(all_results.keys())
    train_accs_final = [all_results[sl]["final_train_acc"] for sl in seq_lengths]
    test_accs_final = [all_results[sl]["final_test_acc"] for sl in seq_lengths]
    
    ax.plot(seq_lengths, train_accs_final, label='Train Accuracy', 
            marker='o', linewidth=2, markersize=8)
    ax.plot(seq_lengths, test_accs_final, label='Test Accuracy', 
            marker='s', linewidth=2, markersize=8)
    ax.set_xlabel('Sequence Length (timesteps)', fontsize=12)
    ax.set_ylabel('Classification Accuracy', fontsize=12)
    ax.set_title(f'{task.capitalize()} Classification Accuracy vs Sequence Length', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.set_xscale('log')
    ax.set_xticks(seq_lengths)
    ax.set_xticklabels([str(sl) for sl in seq_lengths])
    
    plt.tight_layout()
    plt.savefig(f'plots/{task}_accuracy_vs_sequence_length.pdf')
    plt.close()
    print(f"\nComparison plot saved to plots/{task}_accuracy_vs_sequence_length.pdf")
    
    # Print summary table
    print("\n" + "=" * 80)
    print(f"SUMMARY: {task.upper()} Classification Accuracy vs Sequence Length")
    print("=" * 80)
    print(f"{'Seq Length':<15} {'Train Acc':<15} {'Test Acc':<15}")
    print("-" * 80)
    for sl in seq_lengths:
        train_acc = all_results[sl]["final_train_acc"]
        test_acc = all_results[sl]["final_test_acc"]
        print(f"{sl:<15} {train_acc:<15.4f} {test_acc:<15.4f}")
    print("=" * 80)


