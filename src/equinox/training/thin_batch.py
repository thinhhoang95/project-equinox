# ********************************************************
# This script will aggregate all transitions inside a batch folder
# to thin flight by flight, use `thin_batch_indv.py`.
# ********************************************************


import os
import pickle
import argparse
from glob import glob
from tqdm import tqdm

from equinox.config import RunConfiguration
from equinox.dp.trespass.thinning import thin_closures
from equinox.dp.trespass.tres_forward import save_transitions

def process_batch(batch_dir: str, output_filename: str, config: RunConfiguration, components: dict):
    """
    Aggregates and thins backward pass transitions from a single batch directory.

    Args:
        batch_dir (str): Path to the batch directory (e.g., '.../tres_runs/batch0').
        output_filename (str): Name of the output file.
        config (RunConfiguration): The run configuration.
        components (dict): Dictionary of shared components (graph, models, etc.).
    """
    backward_pass_files = glob(os.path.join(batch_dir, "BW_*.pkl"))
    # Exclude macOS ._ files
    backward_pass_files = [f for f in backward_pass_files if not f.endswith('._')]

    if not backward_pass_files:
        print(f"No backward pass files (BW_*.pkl) found in {batch_dir}. Skipping.")
        return

    all_transitions = []
    seen_transitions = set()  # Track unique transitions
    total_transitions_processed = 0
    
    print(f"Aggregating {len(backward_pass_files)} files from {batch_dir}...")
    for f_path in tqdm(backward_pass_files, desc=f"Processing {os.path.basename(batch_dir)}"):
        with open(f_path, "rb") as f:
            try:
                transitions = pickle.load(f)
                if isinstance(transitions, list):
                    total_transitions_processed += len(transitions)
                    for transition in transitions:
                        # Convert transition to tuple to make it hashable for set operations
                        transition_tuple = tuple(transition) if not isinstance(transition, tuple) else transition
                        if transition_tuple not in seen_transitions:
                            seen_transitions.add(transition_tuple)
                            all_transitions.append(transition)
            except (pickle.UnpicklingError, EOFError) as e:
                print(f"Warning: Could not load or file is empty: {f_path}. Error: {e}")

    duplicates_removed = total_transitions_processed - len(all_transitions)
    print(f"Aggregated {len(all_transitions)} unique transitions (removed {duplicates_removed} duplicates). Now thinning...")

    # Thin the closures
    G = components['graph']
    node_to_idx = components['node_to_idx']
    
    # Assuming origin/goal are consistent for the case.
    source_node_idx = node_to_idx[config.origin_node]
    goal_node_idx = node_to_idx[config.goal_node]

    # Option A: infer max_rho directly from the closure tuples.
    thinned_transitions = thin_closures(
        source_node_idx,
        goal_node_idx,
        None,
        G,
        all_transitions,
        wallclock_time_bin_k_tolerance_s=config.delta_t_seconds,
        delta_t_seconds_wall_clock=config.delta_t_seconds,
        include_wait_edges_in_output=True,
    )
    
    print(f"Thinned down to {len(thinned_transitions)} transitions.")

    output_path = os.path.join(batch_dir, output_filename)
    with open(output_path, "wb") as f:
        pickle.dump(thinned_transitions, f)

    print(f"Saved {len(thinned_transitions)} thinned transitions to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and thin backward TRES pass transitions from all batch folders."
    )
    parser.add_argument(
        "--config", 
        required=False, 
        default="D:\\project-equinox\\data\\cases\\LEMD_EGLL\\default.yaml",
        help="Path to the base YAML configuration file."
    )
    parser.add_argument(
        "--tres-runs-dir",
        required=False,
        default="D:\\project-equinox\\data\\cases\\LEMD_EGLL\\tres_runs",
        help="The base directory containing the batch folders (e.g., '.../tres_runs').",
    )
    parser.add_argument(
        "--output-filename",
        default="thinned_transitions.pkl",
        help="The name for the output thinned transitions file in each batch folder.",
    )
    args = parser.parse_args()

    # Load configuration and components
    print(f"Loading configuration from {args.config}...")
    try:
        config = RunConfiguration.load_from_yaml(args.config)
        components = config.initialize_all_components()
        print("Configuration and components loaded successfully.")
    except Exception as e:
        print(f"Error loading configuration or initializing components: {e}")
        return

    if not os.path.isdir(args.tres_runs_dir):
        print(f"Error: Directory not found at {args.tres_runs_dir}")
        return

    batch_dirs = sorted([d.path for d in os.scandir(args.tres_runs_dir) if d.is_dir() and d.name.startswith("batch")])

    if not batch_dirs:
        print(f"No batch directories found in {args.tres_runs_dir}. Exiting.")
        return

    print(f"Found {len(batch_dirs)} batch directories to process.")
    for batch_dir in batch_dirs:
        process_batch(batch_dir, args.output_filename, config, components)


if __name__ == "__main__":
    main()
