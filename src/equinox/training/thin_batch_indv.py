import os
import pickle
import glob
import argparse
from equinox.config import RunConfiguration
from equinox.dp.trespass.thinning import thin_closures
from equinox.dp.trespass.tres_forward import save_transitions

def thin_batch(batch_dir: str, config_path: str):
    """
    Loads all backward closures in a directory, applies thinning, and saves the results.

    Args:
        batch_dir: The directory containing batch subdirectories with backward closure files.
        config_path: Path to the configuration YAML file.
    """
    print(f"Loading configuration from {config_path}")
    config = RunConfiguration.load_from_yaml(config_path)
    
    print("Initializing components...")
    components = config.initialize_all_components()
    
    G = components['graph']
    node_to_idx = components['node_to_idx']
    
    # These might be flight-specific if your batch contains different routes.
    # Here we assume they are constant from the main config.
    origin_node_idx = node_to_idx[config.origin_node]
    goal_node_idx = node_to_idx[config.goal_node]

    search_pattern = os.path.join(batch_dir, '**', 'BW_*.pkl')
    print(f"Searching for files with pattern: {search_pattern}")
    
    closure_files = glob.glob(search_pattern, recursive=True)
    
    if not closure_files:
        print("No backward closure files starting with 'BW_' found.")
        return

    print(f"Found {len(closure_files)} backward closure files to process.")

    for filepath in closure_files:
        try:
            print(f"Processing {filepath}...")
            
            with open(filepath, "rb") as f:
                closure_list = pickle.load(f)
            
            print(f"  Loaded {len(closure_list)} closures.")

            # Option A: infer max_rho directly from the closure tuples.
            thinned_closures = thin_closures(
                origin_node_idx,
                goal_node_idx,
                None,
                G,
                closure_list,
                wallclock_time_bin_k_tolerance_s=config.delta_t_seconds,
                delta_t_seconds_wall_clock=config.delta_t_seconds,
                include_wait_edges_in_output=True,
            )
            
            print(f"  Thinned to {len(thinned_closures)} closures.")

            dirname = os.path.dirname(filepath)
            basename = os.path.basename(filepath)
            
            output_basename = basename.replace('BW_', 'CLSR_', 1)
            output_basename_without_ext = os.path.splitext(output_basename)[0]
            
            print(f"  Saving thinned closures to {os.path.join(dirname, output_basename)}")
            save_transitions(thinned_closures, dirname, output_basename_without_ext)

        except Exception as e:
            print(f"  Failed to process {filepath}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Thin backward closures in batch directories.")
    parser.add_argument("--batch_dir", type=str, default="data/cases/LEMD_EGLL/", help="Path to the main directory containing batch subdirectories.", required=False)
    parser.add_argument("--config", type=str, default="data/cases/LEMD_EGLL/default.yaml", help="Path to the configuration YAML file.")
    
    args = parser.parse_args()
    
    thin_batch(args.batch_dir, args.config)
