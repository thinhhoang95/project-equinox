# *************************************************************
# Run test_trespass_preferences_config.py trespasses first!
# Only tres_forward and tres_backward and thinning are needed
# *************************************************************

import os
import pickle
import torch
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from equinox.config import RunConfiguration
from equinox.dp.trespass.amorwin.forward_svi_log_temp import forward_soft_value_iteration
from equinox.dp.trespass.amorwin.backward_svi_log_cost_temp import backward_soft_value_iteration
from equinox.dp.trespass.amorwin.backward_gradient import backward_gradient_pass

# These wrappers need to be at the top level to be pickleable for multiprocessing.
def _run_forward_svi(
    state_transitions, avg_tailwind_knots, G, idx_to_node, origin_node_idx, 
    cost_model, num_nodes, num_time_bins_wall_clock, num_rho_bins, num_phases, 
    distance_matrix_d, airspace_charge_matrix_ac, device, gamma
):
    """Wrapper to run forward SVI in a separate process and handle device placement."""
    # Move tensors and model to the target device within the new process
    cost_model.to(device)
    avg_tailwind_knots = avg_tailwind_knots.to(device)
    distance_matrix_d = distance_matrix_d.to(device)
    airspace_charge_matrix_ac = airspace_charge_matrix_ac.to(device)
    
    v_f = forward_soft_value_iteration(
        state_transitions=state_transitions,
        avg_tailwind_knots_per_transition=avg_tailwind_knots,
        G=G,
        idx_to_node=idx_to_node,
        origin_node_idx=origin_node_idx,
        cost_model=cost_model,
        num_nodes=num_nodes,
        num_time_bins_wall_clock=num_time_bins_wall_clock,
        num_rho_bins=num_rho_bins,
        num_phases=num_phases,
        distance_matrix_d=distance_matrix_d,
        airspace_charge_matrix_ac=airspace_charge_matrix_ac,
        device=device,
        gamma=gamma,
        verbose=False
    )
    # Return result on CPU to safely pass back to the main process
    return v_f.cpu()

def _run_backward_svi(
    state_transitions, avg_tailwind_knots, G, idx_to_node, goal_node_idx, 
    cost_model, num_nodes, num_time_bins_wall_clock, num_rho_bins, num_phases, 
    distance_matrix_d, airspace_charge_matrix_ac, device, gamma
):
    """Wrapper to run backward SVI in a separate process and handle device placement."""
    # Move tensors and model to the target device within the new process
    cost_model.to(device)
    avg_tailwind_knots = avg_tailwind_knots.to(device)
    distance_matrix_d = distance_matrix_d.to(device)
    airspace_charge_matrix_ac = airspace_charge_matrix_ac.to(device)
    
    v_b, _ = backward_soft_value_iteration(
        state_transitions=state_transitions,
        avg_tailwind_knots_per_transition=avg_tailwind_knots,
        G=G,
        idx_to_node=idx_to_node,
        goal_node_idx=goal_node_idx,
        cost_model=cost_model,
        num_nodes=num_nodes,
        num_time_bins_wall_clock=num_time_bins_wall_clock,
        num_rho_bins=num_rho_bins,
        num_phases=num_phases,
        distance_matrix_d=distance_matrix_d,
        airspace_charge_matrix_ac=airspace_charge_matrix_ac,
        device=device,
        gamma=gamma,
        verbose=False
    )
    # Return result on CPU to safely pass back to the main process
    return v_b.cpu()

def train_model(
    config_path: str,
    max_iterations: int = 100,
    learning_rate: float = 1e-8,
    convergence_threshold: float = 1e-4,
    verbose: bool = True
):
    """
    Main training loop for the trespass model.

    This function iteratively performs forward and backward soft value iteration (SVI)
    and uses the results to compute gradients for the cost model parameters via
    Maximum Entropy Inverse Reinforcement Learning. The cost model is then updated
    using an optimizer.

    The training process stops when the maximum absolute value of the gradient falls
    below a convergence threshold or when the maximum number of iterations is reached.

    Args:
        config_path (str): Path to the YAML configuration file.
        max_iterations (int): The maximum number of training iterations.
        learning_rate (float): The learning rate for the Adam optimizer.
        convergence_threshold (float): The threshold for the maximum gradient
                                       element to determine convergence.
        verbose (bool): If True, print detailed progress information.
    """
    # 1. SETUP
    if verbose:
        print("--- Initializing Training ---")

    # Load configuration and initialize components
    config = RunConfiguration.load_from_yaml(config_path)
    components = config.initialize_all_components()
    device = components['device']
    cost_model = components['cost_model'].to(torch.float64) # Ensure model is float64
    
    # Ensure cost model preference matrix requires grad
    if hasattr(cost_model, 'preference_matrix_p'):
        cost_model.preference_matrix_p.requires_grad = True
    else:
        # A more general way to handle parameters
        for param in cost_model.parameters():
            param.requires_grad = True

    # Setup optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, cost_model.parameters()), 
        lr=learning_rate
    )
    
    if verbose:
        print(f"Using device: {device}")
        num_trainable_params = sum(p.numel() for p in cost_model.parameters() if p.requires_grad)
        print(f"Cost model has {num_trainable_params} trainable parameters.")

    # 2. DATA LOADING
    if verbose:
        print("\n--- Loading Pre-computed Data ---")

    # Load thinned transitions
    thinned_transitions_path = os.path.join(config.output_dir, config.thinning_output_file_name + ".pkl")
    try:
        with open(thinned_transitions_path, "rb") as f:
            transitions = pickle.load(f)
        if verbose:
            print(f"Loaded {len(transitions)} transitions from {thinned_transitions_path}")
    except FileNotFoundError:
        print(f"ERROR: Transitions file not found at {thinned_transitions_path}.")
        print("Please run the pre-computation steps (tres passes, thinning) first.")
        return

    # Load pre-computed wind averages
    wind_avg_file_path = config.wind_avg_file_path
    try:
        avg_tailwind_knots = torch.load(wind_avg_file_path).to(device)
        if verbose:
            print(f"Loaded pre-computed wind averages from {wind_avg_file_path}")
    except FileNotFoundError:
        print(f"ERROR: Wind averages file not found at {wind_avg_file_path}.")
        print("Please run `amortize_wind_average` first.")
        return
        
    # Load empirical counts
    empirical_counts_path = os.path.join("data/empirical_counts", f"{config.file_prefix}_empirical_counts.pt")
    try:
        empirical_counts = torch.load(empirical_counts_path).to(device, dtype=torch.float64)
        if verbose:
            print(f"Loaded empirical counts from {empirical_counts_path}")
    except FileNotFoundError:
        print(f"Warning: Empirical counts not found at {empirical_counts_path}.")
        # raise FileNotFoundError(f"Empirical counts not found at {empirical_counts_path}.")
        # print("Using zero counts for all links.")
        # num_nodes = components['num_nodes']
        # empirical_counts = torch.zeros((num_nodes, num_nodes), device=device, dtype=torch.float64)


    # 3. DERIVE DIMENSIONS AND PREPARE TENSORS
    if not transitions:
        print("ERROR: No transitions loaded. Cannot proceed.")
        return
        
    max_k_val = max(max(t[1] for t in transitions), max(t[6] for t in transitions))
    max_rho_val = max(max(t[2] for t in transitions), max(t[7] for t in transitions))
    max_phase_val = max(max(t[4] for t in transitions), max(t[9] for t in transitions))

    num_time_bins_wall_clock = max_k_val + 1
    num_rho_bins = max_rho_val + 1
    num_phases = max_phase_val + 1
    num_nodes = components['num_nodes']
    
    if verbose:
        print("\n--- Derived State Space Dimensions ---")
        print(f"Num nodes: {num_nodes}")
        print(f"Num time bins: {num_time_bins_wall_clock}")
        print(f"Num rho bins: {num_rho_bins}")
        print(f"Num phases: {num_phases}")

    dist_matrix = torch.tensor(components['dist_matrix'], dtype=torch.float64, device=device)
    ac_matrix = torch.tensor(components['ac_matrix'], dtype=torch.float64, device=device)

    # Create CPU copies of large constant tensors to pass to worker processes
    dist_matrix_cpu = dist_matrix.cpu()
    ac_matrix_cpu = ac_matrix.cpu()
    avg_tailwind_knots_cpu = avg_tailwind_knots.cpu()


    # 4. TRAINING LOOP
    if verbose:
        print("\n--- Starting Training Loop ---")

    with ProcessPoolExecutor(max_workers=2) as executor:
        # Create progress bar with custom format
        pbar = tqdm(range(max_iterations), 
                   desc="Training", 
                   unit="iter",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
        
        for i in pbar:
            iter_time_start = time.time()
            if verbose:
                print(f"\n--- Iteration {i+1}/{max_iterations} ---")
            
            # Move model to CPU for safe pickling to worker processes
            cost_model_cpu = cost_model.to('cpu')

            # a/b. Forward and Backward SVI Passes (in parallel)
            future_fwd = executor.submit(
                _run_forward_svi,
                transitions,
                avg_tailwind_knots_cpu,
                components['graph'],
                components['idx_to_node'],
                components['origin_node_idx'],
                cost_model_cpu,
                num_nodes,
                num_time_bins_wall_clock,
                num_rho_bins,
                num_phases,
                dist_matrix_cpu,
                ac_matrix_cpu,
                device,
                config.gamma
            )
            
            future_bwd = executor.submit(
                _run_backward_svi,
                transitions,
                avg_tailwind_knots_cpu,
                components['graph'],
                components['idx_to_node'],
                components['goal_node_idx'],
                cost_model_cpu,
                num_nodes,
                num_time_bins_wall_clock,
                num_rho_bins,
                num_phases,
                dist_matrix_cpu,
                ac_matrix_cpu,
                device,
                config.gamma
            )

            # Retrieve results and move them to the target device
            V_f = future_fwd.result().to(device)
            V_b = future_bwd.result().to(device)
            
            # c. Gradient Calculation
            _, grad = backward_gradient_pass(
                state_transitions=transitions,
                avg_tailwind_knots_per_transition=avg_tailwind_knots,
                V_f=V_f,
                V_b=V_b,
                cost_model=cost_model,
                empirical_counts=empirical_counts,
                origin_node_idx=components['origin_node_idx'],
                num_nodes=num_nodes,
                distance_matrix_d=dist_matrix,
                airspace_charge_matrix_ac=ac_matrix,
                device=device,
                gamma=config.gamma,
                verbose=False
            )

            # d. Optimizer Step
            optimizer.zero_grad()
            
            param_idx = 0
            for param in cost_model.parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        param.grad.zero_()
                    num_param_elements = param.numel()
                    grad_slice = grad[param_idx : param_idx + num_param_elements].view(param.shape).to(param.dtype)
                    param.grad = grad_slice
                    param_idx += num_param_elements
            
            optimizer.step()

            # e. Logging and Convergence Check
            max_grad = torch.max(torch.abs(grad)).item()
            iter_time_end = time.time()
            iter_duration = iter_time_end - iter_time_start

            # Update progress bar with current metrics
            pbar.set_postfix({
                'max_grad': f'{max_grad:.2e}',
                'iter_time': f'{iter_duration:.1f}s',
                'threshold': f'{convergence_threshold:.1e}'
            })

            if verbose:
                print(f"Iteration finished in {iter_duration:.2f} seconds.")
                print(f"Max absolute gradient: {max_grad:.6f}")

            if max_grad < convergence_threshold:
                pbar.close()
                print(f"\nConvergence reached at iteration {i+1}. Max gradient {max_grad:.6f} is below threshold {convergence_threshold}.")
                break
        else: # This 'else' belongs to the 'for' loop, and runs if the loop completes without a 'break'
            pbar.close()
            print(f"\nMax iterations ({max_iterations}) reached.")

    # 5. POST-TRAINING
    print("\n--- Training Finished ---")
    
    output_dir = os.path.join(config.output_dir, "trained_models")
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, f"{config.file_prefix}_cost_model.pt")

    torch.save(cost_model.state_dict(), model_save_path)
    print(f"Trained cost model saved to: {model_save_path}")

    return cost_model


if __name__ == '__main__':
    # This block allows running the training script directly.
    # You need to ensure the config file and pre-computed data are in the correct paths.
    
    # Example configuration path
    CONFIG_PATH = "data/profiles/nbjet_35450_egll_lemd_2023_04_01.yaml"

    print("Starting training process from __main__...")
    start_time = time.time()
    
    trained_model = train_model(
        config_path=CONFIG_PATH,
        max_iterations=50,
        learning_rate=1e-4,
        convergence_threshold=1e-3,
        verbose=True
    )
    
    end_time = time.time()
    print(f"\nTotal training script execution time: {end_time - start_time:.2f} seconds.")
