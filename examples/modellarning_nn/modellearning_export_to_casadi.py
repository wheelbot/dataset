import casadi as cs
import numpy as np
import jax.numpy as jnp
import equinox as eqx
import pickle
import json
from pathlib import Path
from modellearning_common import load_dynamics_model

import csnn

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
CASADI_SYM_TYPE = "MX"  # Can be either "SX" or "MX"
csnn.set_sym_type(CASADI_SYM_TYPE)

# Get the appropriate CasADi symbolic type
if CASADI_SYM_TYPE == "SX":
    casadi_sym = cs.SX
elif CASADI_SYM_TYPE == "MX":
    casadi_sym = cs.MX
else:
    raise ValueError(f"Invalid CASADI_SYM_TYPE: {CASADI_SYM_TYPE}. Must be 'SX' or 'MX'.")
# ============================================================================

def extract_mlp_structure(model):
    """
    Extract layer structure and parameters from an Equinox MLP model.
    
    Args:
        model: Equinox MLP model
        
    Returns:
        Tuple of (layer_info, parameters) where:
        - layer_info: Dict mapping layer index to layer type and dimensions
        - parameters: Dict mapping parameter names to JAX arrays
    """
    # Check that model is an Equinox MLP
    if not isinstance(model, eqx.nn.MLP):
        raise TypeError(f"Model must be an eqx.nn.MLP, got {type(model)}")
    
    layer_info = {}
    parameters = {}
    
    # Get activation function name
    activation_name = model.activation.fun.__name__
    
    # Map JAX activation names to csnn layer types
    activation_map = {
        'relu': 'ReLU',
        'leaky_relu': 'LeakyReLU',
        'tanh': 'Tanh',
        'sigmoid': 'Sigmoid',
        'elu': 'ELU',
        'gelu': 'GELU',
        'softplus': 'Softplus',
    }
    
    if activation_name not in activation_map:
        raise NotImplementedError(f"Activation '{activation_name}' is not supported. "
                                  f"Supported activations: {list(activation_map.keys())}")
    
    activation_type = activation_map[activation_name]
    
    # Equinox MLP structure: model.layers is a list of Linear layers only
    # Activations are applied between layers (not after the last layer)
    layer_idx = 0
    num_linear_layers = len(model.layers)
    
    for i, layer in enumerate(model.layers):
        if not isinstance(layer, eqx.nn.Linear):
            raise TypeError(f"Expected Linear layer at position {i}, got {type(layer)}")
        
        in_features = layer.weight.shape[1]
        out_features = layer.weight.shape[0]
        
        # Add linear layer
        layer_info[str(layer_idx)] = {
            'type': 'Linear',
            'in_features': in_features,
            'out_features': out_features
        }
        
        parameters[f'layer_{layer_idx}_weight'] = layer.weight
        parameters[f'layer_{layer_idx}_bias'] = layer.bias
        
        layer_idx += 1
        
        # Add activation after all layers except the last one
        if i < num_linear_layers - 1:
            layer_info[str(layer_idx)] = {
                'type': activation_type
            }
            layer_idx += 1
    
    return layer_info, parameters


def create_casadi_mlp(layer_info):
    """
    Create a csnn Sequential model from layer information.
    
    Args:
        layer_info: Dict mapping layer index to layer type and dimensions
        
    Returns:
        csnn.Sequential model
    """
    layers = []
    for name in sorted(layer_info.keys(), key=int):
        ltype = layer_info[name]['type']

        if ltype == 'Linear':
            in_f = layer_info[name]['in_features']
            out_f = layer_info[name]['out_features']
            layers.append(csnn.Linear(in_f, out_f))

        elif ltype == 'LeakyReLU':
            layers.append(csnn.LeakyReLU())

        elif ltype == 'Tanh':
            layers.append(csnn.Tanh())

        elif ltype == 'Softplus':
            layers.append(csnn.Softplus())
            
        elif ltype == 'ELU':
            layers.append(csnn.ELU())
            
        elif ltype == 'ReLU':
            layers.append(csnn.ReLU())
        
        elif ltype == 'GELU':
            layers.append(csnn.GELU())

        else:
            raise NotImplementedError(f"Layer type '{ltype}' is not implemented in this loader.")

    model = csnn.Sequential[casadi_sym](layers)
    return model


def set_casadi_model_parameters(casadi_model, jax_parameters):
    """
    Set the parameters of a CasADi model from JAX parameters.
    
    Args:
        casadi_model: csnn.Sequential model
        jax_parameters: Dict of JAX parameter arrays
        
    Returns:
        Dict of CasADi parameter values
    """
    casadi_params = dict(casadi_model.parameters(prefix="nn"))
    param_values = {}
    
    # Map JAX parameters to CasADi parameters
    for key in sorted([k for k in jax_parameters.keys() if 'weight' in k]):
        # Extract layer index from key format "layer_{layer_idx}_weight"
        layer_idx = int(key.split('_')[1])
        
        weight_key = key
        bias_key = key.replace('weight', 'bias')
        
        # Get JAX arrays
        weight_jax = jax_parameters[weight_key]
        bias_jax = jax_parameters[bias_key]
        
        # Convert to numpy
        weight_np = np.array(weight_jax)
        bias_np = np.array(bias_jax)
        
        # Assign to CasADi parameters
        # csnn names parameters as 'prefix.layer_idx.weight' and 'prefix.layer_idx.bias'
        weight_param_name = f"nn.{layer_idx}.weight"
        bias_param_name = f"nn.{layer_idx}.bias"
        
        if weight_param_name in casadi_params:
            param_values[weight_param_name] = weight_np
        if bias_param_name in casadi_params:
            param_values[bias_param_name] = bias_np
            
    
    return param_values


def build_casadi_dynamics_function(model_path):
    """
    Load a trained dynamics model and build a CasADi function.
    
    The resulting function takes state and action as inputs, normalizes them,
    applies the neural network to predict state deltas, adds them to the state,
    and denormalizes the result.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Tuple of (casadi_function, param_dict) where:
        - casadi_function: CasADi Function with signature (state, action, params) -> next_state
        - param_dict: Dict of parameter names to values (weights and normalization params)
    """
    # Load the model
    model, hyperparams = load_dynamics_model(model_path)
    
    # Extract normalization parameters
    state_mean = jnp.array(hyperparams["state_mean"])
    state_std = jnp.array(hyperparams["state_std"])
    action_mean = jnp.array(hyperparams["action_mean"])
    action_std = jnp.array(hyperparams["action_std"])
    state_dim = hyperparams["state_dim"]
    action_dim = hyperparams["action_dim"]
    
    print(f"Loading model from {model_path}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # Extract MLP structure and parameters
    layer_info, jax_params = extract_mlp_structure(model)
    
    print(f"Model structure: {len(layer_info)} layers")
    for idx, info in layer_info.items():
        print(f"  Layer {idx}: {info}")
    
    # Create CasADi model
    casadi_model = create_casadi_mlp(layer_info)
    
    # Get CasADi parameters
    casadi_params = dict(casadi_model.parameters(prefix="nn"))
    param_values = set_casadi_model_parameters(casadi_model, jax_params)
    
    # Create CasADi symbolic variables for inputs
    state_sym = casadi_sym.sym('state', state_dim)
    action_sym = casadi_sym.sym('action', action_dim)
    
    # Create CasADi parameters for normalization
    state_mean_sym = casadi_sym.sym('state_mean', state_dim)
    state_std_sym = casadi_sym.sym('state_std', state_dim)
    action_mean_sym = casadi_sym.sym('action_mean', action_dim)
    action_std_sym = casadi_sym.sym('action_std', action_dim)
    
    # Normalize inputs
    state_norm = (state_sym - state_mean_sym) / state_std_sym
    action_norm = (action_sym - action_mean_sym) / action_std_sym
    
    # Concatenate normalized state and action
    model_input = cs.vertcat(state_norm, action_norm)
    
    # Apply neural network to get delta state (normalized)
    delta_state_norm = casadi_model(model_input.T).T
    
    # Add delta to normalized state
    next_state_norm = state_norm + delta_state_norm
    
    # Denormalize next state
    next_state = next_state_norm * state_std_sym + state_mean_sym
    
    # Create CasADi function with separate parameters for each NN weight/bias and normalization
    # Build input list: state, action, then all NN parameters, then normalization parameters
    input_list = [state_sym, action_sym]
    input_names = ['state', 'action']
    
    # Add each NN parameter as a separate input
    for param_name in sorted(casadi_params.keys()):
        param = casadi_params[param_name]
        input_list.append(param)
        input_names.append(param_name)
    
    # Add normalization parameters as separate inputs
    input_list.extend([state_mean_sym, state_std_sym, action_mean_sym, action_std_sym])
    input_names.extend(['state_mean', 'state_std', 'action_mean', 'action_std'])
    
    casadi_fn = cs.Function(
        'dynamics_model',
        input_list,
        [next_state],
        input_names,
        ['next_state']
    )
    
    # Create parameter dictionary for reference
    param_dict = {
        'nn_params': param_values,
        'state_mean': state_mean,
        'state_std': state_std,
        'action_mean': action_mean,
        'action_std': action_std,
    }
    
    total_nn_params = sum(p.size for p in param_values.values())
    print(f"\nCasADi function created:")
    print(f"  Inputs: state ({state_dim}), action ({action_dim}), {len(param_values)} NN params, 4 normalization params")
    print(f"  Output: next_state ({state_dim})")
    print(f"  Total NN parameters: {total_nn_params}")
    
    return casadi_fn, param_dict


def export_model_to_casadi(model_path, tolerance=1e-4):
    """
    Export a trained dynamics model to CasADi format with validation.
    
    This function:
    1. Converts the JAX/Equinox model to CasADi
    2. Tests numerical equivalence between JAX and CasADi predictions
    3. Exports the model in three formats:
       - .casadi: CasADi function serialization
       - .pkl: Python pickle with parameters and test data
       - .json: Human-readable JSON with metadata
    
    Args:
        model_path: Path to the trained model file (.eqx)
        tolerance: Maximum allowed difference between JAX and CasADi predictions (default: 1e-6)
        
    Returns:
        Dict with paths to exported files
        
    Raises:
        ValueError: If numerical validation fails (predictions differ by more than tolerance)
    """
    model_path = Path(model_path)
    base_path = model_path.with_suffix('')
    
    print("=" * 70)
    print(f"Exporting model: {model_path.name}")
    print("=" * 70)
    
    # Step 1: Convert to CasADi
    print("\n[1/4] Converting JAX model to CasADi...")
    casadi_fn, param_dict = build_casadi_dynamics_function(str(model_path))
    
    # Step 2: Generate random test inputs
    print("\n[2/4] Generating random test inputs...")
    state_dim = len(param_dict['state_mean'])
    action_dim = len(param_dict['action_mean'])
    
    np.random.seed(42)  # Fixed seed for reproducible test
    test_state = np.random.randn(state_dim)
    test_action = np.random.randn(action_dim)
    
    print(f"  State shape: {test_state.shape}")
    print(f"  Action shape: {test_action.shape}")
    
    # Step 3: Test numerical equivalence
    print("\n[3/4] Validating numerical equivalence...")
    
    # Prepare arguments for CasADi function
    fn_args = [test_state, test_action]
    for param_name in sorted(param_dict['nn_params'].keys()):
        fn_args.append(param_dict['nn_params'][param_name])
    fn_args.extend([
        param_dict['state_mean'],
        param_dict['state_std'],
        param_dict['action_mean'],
        param_dict['action_std']
    ])
    
    # Evaluate CasADi function
    next_state_casadi = casadi_fn(*fn_args)
    next_state_casadi_np = np.array(next_state_casadi).flatten()
    
    # Load and evaluate JAX model
    jax_model, jax_hyperparams = load_dynamics_model(str(model_path))
    
    state_mean_jax = jnp.array(jax_hyperparams["state_mean"])
    state_std_jax = jnp.array(jax_hyperparams["state_std"])
    action_mean_jax = jnp.array(jax_hyperparams["action_mean"])
    action_std_jax = jnp.array(jax_hyperparams["action_std"])
    
    test_state_jax = jnp.array(test_state)
    test_action_jax = jnp.array(test_action)
    
    state_norm_jax = (test_state_jax - state_mean_jax) / state_std_jax
    action_norm_jax = (test_action_jax - action_mean_jax) / action_std_jax
    
    model_input_jax = jnp.concatenate([state_norm_jax, action_norm_jax])
    delta_state_norm_jax = jax_model(model_input_jax)
    next_state_norm_jax = state_norm_jax + delta_state_norm_jax
    next_state_jax = next_state_norm_jax * state_std_jax + state_mean_jax
    next_state_jax_np = np.array(next_state_jax).flatten()
    
    # Compare predictions
    difference = np.abs(next_state_casadi_np - next_state_jax_np)
    max_diff = np.max(difference)
    mean_diff = np.mean(difference)
    
    print(f"  Max absolute difference:  {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    print(f"  Tolerance threshold:      {tolerance:.2e}")
    
    if max_diff >= tolerance:
        raise ValueError(
            f"Numerical validation failed! "
            f"Max difference {max_diff:.2e} exceeds tolerance {tolerance:.2e}"
        )
    
    print("  ✓ Validation passed!")
    
    # Step 4: Export to multiple formats
    print("\n[4/4] Exporting model files...")
    
    # 4a: Export CasADi function (.casadi)
    casadi_path = base_path.with_suffix('.casadi')
    casadi_fn.save(str(casadi_path))
    print(f"  ✓ CasADi function: {casadi_path.name}")
    
    # 4b: Export parameters and test data (.pkl)
    pkl_data = {
        'nn_params': {k: np.array(v) for k, v in param_dict['nn_params'].items()},
        'state_mean': np.array(param_dict['state_mean']),
        'state_std': np.array(param_dict['state_std']),
        'action_mean': np.array(param_dict['action_mean']),
        'action_std': np.array(param_dict['action_std']),
        'test_inputs': {
            'state': test_state,
            'action': test_action
        },
        'test_outputs': {
            'next_state_casadi': next_state_casadi_np,
            'next_state_jax': next_state_jax_np
        },
        'validation': {
            'max_difference': float(max_diff),
            'mean_difference': float(mean_diff),
            'tolerance': tolerance
        }
    }
    
    pkl_path = base_path.with_suffix('.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(pkl_data, f)
    print(f"  ✓ Pickle data:     {pkl_path.name}")
    
    # 4c: Export metadata (.json)
    # Convert NN parameters to nested lists (handles 2D arrays for weights, 1D for biases)
    nn_params_json = {}
    for param_name, param_value in param_dict['nn_params'].items():
        nn_params_json[param_name] = np.array(param_value).tolist()
    
    json_data = {
        'model_info': {
            'source_file': str(model_path.name),
            'state_dim': int(state_dim),
            'action_dim': int(action_dim),
            'state_labels': jax_hyperparams.get('states_labels', []),
            'action_labels': jax_hyperparams.get('actions_labels', [])
        },
        'nn_parameters': nn_params_json,
        'normalization': {
            'state_mean': param_dict['state_mean'].tolist(),
            'state_std': param_dict['state_std'].tolist(),
            'action_mean': param_dict['action_mean'].tolist(),
            'action_std': param_dict['action_std'].tolist()
        },
        'network_structure': {
            'num_parameters': sum(p.size for p in param_dict['nn_params'].values()),
            'parameter_names': list(param_dict['nn_params'].keys())
        },
        'test_data': {
            'inputs': {
                'state': test_state.tolist(),
                'action': test_action.tolist()
            },
            'outputs': {
                'next_state_casadi': next_state_casadi_np.tolist(),
                'next_state_jax': next_state_jax_np.tolist()
            }
        },
        'validation': {
            'max_difference': float(max_diff),
            'mean_difference': float(mean_diff),
            'tolerance': tolerance
        }
    }
    
    json_path = base_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  ✓ JSON metadata:   {json_path.name}")
    
    print("\n" + "=" * 70)
    print("Export complete!")
    print("=" * 70)
    
    return {
        'casadi': casadi_path,
        'pickle': pkl_path,
        'json': json_path
    }


if __name__ == "__main__":
    # Example usage
    model_path = "models/trained_model_multistep.eqx"
    
    try:
        # Export model to CasADi
        exported_files = export_model_to_casadi(model_path, tolerance=5e-4)
        
        print("\nExported files:")
        for format_type, path in exported_files.items():
            print(f"  {format_type:10s}: {path}")
        
    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        raise


