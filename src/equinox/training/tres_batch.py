import pandas as pd
import os
from datetime import datetime
from copy import deepcopy
import logging
from tqdm import tqdm
import argparse
import multiprocessing
from functools import partial
import torch
import networkx as nx
import csv
import time
import random

from equinox.wind.batch_wind_model import get_flight_batches
from equinox.wind.wind_date import WindDate
from equinox.config import RunConfiguration
from equinox.dp.trespass.tres_forward import tres_forward, save_transitions
from equinox.dp.trespass.tres_backward import tres_backward
from equinox.dp.trespass.thinning import thin_closures
from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight
from equinox.training.prep.resculpt_viterbi import viterbi_match, haversine_nm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def initialize_csv_file(csv_path, fieldnames):
    """Initialize the CSV file with headers."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


def write_csv_row_safe(csv_path, fieldnames, row_dict, max_retries=10):
    """Write a single row to the CSV file in a process-safe manner using retry logic."""
    for attempt in range(max_retries):
        try:
            with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(row_dict)
            return  # Success, exit the function
        except (OSError, IOError) as e:
            if attempt < max_retries - 1:
                # Wait a random short time before retrying to reduce collision probability
                time.sleep(random.uniform(0.01, 0.05))
                continue
            else:
                # Final attempt failed, log the error
                logging.error(f"Failed to write to CSV after {max_retries} attempts: {e}")
                raise


def process_flight(flight_series, config, components, case_name, batch_idx, output_dir_base, csv_path=None, csv_fieldnames=None):
    """
    Process a single flight: run TRES forward and backward passes.
    This version creates its own wind model for the flight.

    Args:
        flight_series (pd.Series): A row from the flight DataFrame.
        config (RunConfiguration): The base configuration.
        components (dict): A dictionary of shared components (graph, models, etc.), without the wind model.
        case_name (str): The name of the case (e.g., 'LEMD_EGLL').
        batch_idx (int): The index of the current batch.
        output_dir_base (str): The root directory for outputs.
        csv_path (str, optional): Path to the CSV file for snapped routes.
        csv_fieldnames (list, optional): Field names for the CSV file.
    """
    flight_id = flight_series['flight_id']
    takeoff_ts = flight_series['takeoff_time']

    # Prepare output directory and check for existing files
    output_dir = os.path.join(output_dir_base, f"batch{batch_idx}")
    fw_filename_pkl = os.path.join(output_dir, f"FW_{flight_id}_{takeoff_ts}.pkl")
    bw_filename_pkl = os.path.join(output_dir, f"BW_{flight_id}_{takeoff_ts}.pkl")
    clsr_filename_pkl = os.path.join(output_dir, f"CLSR_{flight_id}_{takeoff_ts}.pkl")
    wind_filename_pt = os.path.join(output_dir, f"WIND_{flight_id}_{takeoff_ts}.pt")

    if all(os.path.exists(f) for f in [fw_filename_pkl, bw_filename_pkl, clsr_filename_pkl, wind_filename_pt]):
        logging.info(f"Skipping already processed flight {flight_id}.")
        return

    # This log might be helpful for debugging in a multiprocessing context
    # logging.info(f"Processing flight {flight_id} with takeoff {takeoff_ts}")

    # 1. Prepare flight-specific configuration and create a dedicated wind model
    flight_config = deepcopy(config)
    flight_config.origin_node = flight_series['origin']
    flight_config.goal_node = flight_series['destination']

    dt_takeoff = datetime.fromtimestamp(takeoff_ts)
    dt_landing = datetime.fromtimestamp(flight_series['landing_time'])

    time_format = '%Y-%m-%d %H:%M:%S'
    flight_config.takeoff_time_str = dt_takeoff.strftime(time_format)
    flight_config.landing_time_str = dt_landing.strftime(time_format)
    flight_config.estimated_takeoff_time_str = flight_config.takeoff_time_str
    flight_config.estimated_landing_time_str = flight_config.landing_time_str
    
    flight_components = components.copy()
    try:
        # Create a WindDate model for the specific flight date
        date_str = dt_takeoff.strftime('%Y-%m-%d')
        flight_wind_model = WindDate(
            date_str=date_str,
            data_dir=flight_config.wind_data_dir
        )
        flight_components['wind_model'] = flight_wind_model
    except Exception as e:
        logging.error(f"Failed to create WindDate for flight {flight_id} on date {date_str}: {e}", exc_info=True)
        import sys
        sys.exit(1)
        return  # Skip this flight if wind model initialization fails

    # Prepare output directory and filenames
    output_dir = os.path.join(output_dir_base, f"batch{batch_idx}")
    os.makedirs(output_dir, exist_ok=True)

    fw_filename = f"FW_{flight_id}_{takeoff_ts}"
    bw_filename = f"BW_{flight_id}_{takeoff_ts}"

    # 2. Run TRES Forward Pass
    transitions_list = None
    try:
        _, _, _, transitions_list = tres_forward(
            graph=flight_components['graph'],
            source_node_id=flight_config.origin_node,
            takeoff_time_str=flight_config.takeoff_time_str,
            source_elevation_ft=flight_config.source_elevation_ft,
            goal_elevation_ft=flight_config.goal_elevation_ft,
            cost_model=flight_components['cost_model'],
            wind_model=flight_components['wind_model'],
            performance_model=flight_components['performance_model'],
            dist_matrix_np=flight_components['dist_matrix'],
            ac_matrix_np=flight_components['ac_matrix'],
            initial_alt_ft=flight_config.initial_alt_ft,
            delta_t_seconds=flight_config.delta_t_seconds,
            max_flight_duration_hours=flight_config.max_flight_duration_hours,
            etto_delta_t_seconds=flight_config.etto_delta_t_seconds,
            max_elapsed_time_since_takeoff_hours=flight_config.max_elapsed_time_since_takeoff_hours,
            device=flight_components['device']
        )
        save_transitions(transitions_list, output_dir, fw_filename)
        # logging.info(f"Forward TRES for {flight_id} completed.")
    except Exception as e:
        logging.error(f"Error during forward TRES for flight {flight_id}: {e}", exc_info=True)
        return  # Skip backward pass if forward fails
    
    if transitions_list is None or not transitions_list:
        logging.warning(f"Forward TRES for {flight_id} produced no transitions. Skipping backward pass.")
        return

    # 3. Run TRES Backward Pass
    try:
        state_closure_list = tres_backward(
            graph=flight_components['graph'],
            goal_node_id=flight_config.goal_node,
            estimated_landing_time_str=flight_config.estimated_landing_time_str,
            origin_elevation_ft=flight_config.source_elevation_ft,
            destination_elevation_ft=flight_config.goal_elevation_ft,
            wind_model=flight_components['wind_model'],
            performance_model=flight_components['performance_model'],
            transitions_list=transitions_list,
            eta_takeoff_str=flight_config.estimated_takeoff_time_str,
            final_alt_ft=flight_config.goal_elevation_ft,
            delta_t_seconds_wall_clock=flight_config.delta_t_seconds,
            delta_t_seconds_climb=flight_config.etto_delta_t_seconds,
            max_flight_duration_hours=flight_config.max_flight_duration_hours,
            climb_phase_switch_allowance_climb_time_bins=flight_config.climb_phase_switch_allowance_climb_time_bins,
            device=flight_components['device']
        )
        save_transitions(state_closure_list, output_dir, bw_filename)
        # logging.info(f"Backward TRES for {flight_id} completed.")
    except Exception as e:
        logging.error(f"Error during backward TRES for flight {flight_id}: {e}", exc_info=True)
        return

    if state_closure_list is None or not state_closure_list:
        logging.warning(f"Backward TRES for {flight_id} produced no closures. Skipping thinning and wind amortization.")
        return

    # 4. Thinning
    logging.info(f"Performing thinning for flight {flight_id}.")
    try:
        node_to_idx = flight_components['node_to_idx']
        source_node_idx = node_to_idx[flight_config.origin_node]
        goal_node_idx = node_to_idx[flight_config.goal_node]
        # Using a fixed max_rho as seen in test files.
        # This parameter is related to the maximum number of climb time bins.
        max_rho = 36
        
        thinned_transitions = thin_closures(
            source_node_idx, goal_node_idx, max_rho,
            flight_components['graph'], state_closure_list
        )
        
        thinned_filename = f"CLSR_{flight_id}_{takeoff_ts}"
        save_transitions(thinned_transitions, output_dir, thinned_filename)
        logging.info(f"Thinned transitions for {flight_id} saved.")

    except Exception as e:
        logging.error(f"Error during thinning for flight {flight_id}: {e}", exc_info=True)
        return

    if not thinned_transitions:
        logging.warning(f"Thinning for {flight_id} produced no transitions. Skipping wind amortization.")
        return None
        
    # 4.5. Route Snapping to Feasible Graph
    logging.info(f"Snapping route to feasible graph for flight {flight_id}.")
    try:
        # Extract feasible waypoint transitions
        waypoint_transitions = extract_feasible_waypoint_transitions(thinned_transitions)
        
        # Create reverse mapping from node index to node name
        idx_to_node = {v: k for k, v in flight_components['node_to_idx'].items()}
        
        # Create feasible graph containing only edges from thinned transitions
        feasible_graph = create_feasible_graph(
            flight_components['graph'], waypoint_transitions, idx_to_node
        )

        # For debugging, save the feasible graph as a PDF plot
        from equinox.helpers.plotters import plot_route_graph_pdf
        # feasible_graph_pdf_filename = f"FEAS_{flight_id}_{takeoff_ts}.pdf"
        # feasible_graph_pdf_filepath = os.path.join(output_dir, feasible_graph_pdf_filename)
        # plot_route_graph_pdf(feasible_graph, show_label=True, highlighted_labels=[], output_path=feasible_graph_pdf_filepath)
        # logging.info(f"Feasible graph PDF for {flight_id} saved to {feasible_graph_pdf_filepath}")
        
        # Get the original route from flight data and snap it to the feasible graph
        original_route_str = flight_series.get('route', '')
        snapped_route_str = snap_route_to_feasible_graph(
            original_route_str, feasible_graph, flight_components['graph']
        )
        
        # Write snapped route information directly to CSV
        if snapped_route_str and csv_path and csv_fieldnames:
            snapped_route_info = {
                "flight_id": flight_id,
                "route": snapped_route_str,
                "takeoff_time": flight_series.get('takeoff_time'),
                "landing_time": flight_series.get('landing_time'),
                "cruise_altitude": flight_series.get('cruise_altitude'),
                "origin": flight_series.get('origin'),
                "destination": flight_series.get('destination'),
                "flight_time_s": flight_series.get('flight_time_s')
            }
            write_csv_row_safe(csv_path, csv_fieldnames, snapped_route_info)
            logging.info(f"Successfully snapped route for flight {flight_id}")
        else:
            logging.warning(f"Failed to snap route for flight {flight_id}")
            
    except Exception as e:
        logging.error(f"Error during route snapping for flight {flight_id}: {e}", exc_info=True)
        
    # 5. Wind Amortization
    logging.info(f"Amortizing wind for flight {flight_id}.")
    try:
        node_coords_deg = flight_components['node_coords_deg']
        # CRITICAL FIX: The 'k' time bins in thinned_transitions are from the backward pass,
        # which uses estimated_landing_ssm - max_flight_duration_hours * 3600 as the time reference.
        # We must use the SAME time reference here to calculate correct wind times.
        estimated_landing_ssm = datestr_to_seconds_since_midnight(flight_config.estimated_landing_time_str)
        min_wall_clock_time_sec = float(estimated_landing_ssm - flight_config.max_flight_duration_hours * 3600)
        
        avg_tailwind_knots = flight_components['wind_model'].get_average_tailwind_on_edges_knots(
            transitions=thinned_transitions,
            node_coords_deg=node_coords_deg,
            min_wall_clock_time_sec=min_wall_clock_time_sec,
            delta_t_wall_clock_sec=flight_config.delta_t_seconds, # This is the wall-clock delta
            num_integration_steps=3
        )
        
        wind_filename = f"WIND_{flight_id}_{takeoff_ts}.pt"
        wind_filepath = os.path.join(output_dir, wind_filename)
        torch.save(avg_tailwind_knots, wind_filepath)
        logging.info(f"Amortized wind for {flight_id} saved to {wind_filepath}")

    except Exception as e:
        logging.error(f"Error during wind amortization for flight {flight_id}: {e}", exc_info=True)


def extract_feasible_waypoint_transitions(thinned_transitions):
    """
    Extract unique waypoint transitions (u_idx, v_idx) from thinned transitions.
    
    Args:
        thinned_transitions: List of tuples with structure:
            (u_idx, k_u, rho_u, alt_u, phase_u, v_idx, k_v, rho_v, alt_v, phase_v)
    
    Returns:
        set: Set of (u_idx, v_idx) tuples representing feasible waypoint transitions
    """
    waypoint_transitions = set()
    for transition in thinned_transitions:
        # Extract u_idx (index 0) and v_idx (index 5) from the transition tuple
        u_idx = transition[0]
        v_idx = transition[5]
        waypoint_transitions.add((u_idx, v_idx))
    return waypoint_transitions


def create_feasible_graph(original_graph, waypoint_transitions, idx_to_node):
    """
    Create a subgraph containing only feasible edges based on waypoint transitions.
    
    Args:
        original_graph: The original NetworkX graph
        waypoint_transitions: Set of (u_idx, v_idx) tuples
        idx_to_node: Dictionary mapping node indices to node names
    
    Returns:
        nx.DiGraph: A new graph with only feasible edges
    """
    feasible_graph = nx.DiGraph()
    
    # Add all nodes from the original graph
    for node, data in original_graph.nodes(data=True):
        feasible_graph.add_node(node, **data)
    
    # Add only feasible edges based on waypoint transitions
    for u_idx, v_idx in waypoint_transitions:
        u_node = idx_to_node.get(u_idx)
        v_node = idx_to_node.get(v_idx)
        
        if u_node is not None and v_node is not None and original_graph.has_edge(u_node, v_node):
            edge_data = original_graph.edges[u_node, v_node]
            feasible_graph.add_edge(u_node, v_node, **edge_data)
    
    return feasible_graph


def snap_route_to_feasible_graph(original_route_str, feasible_graph, original_graph):
    """
    Snap the original route to the feasible graph using viterbi matching.
    
    Args:
        original_route_str: Space-separated string of waypoint names
        feasible_graph: NetworkX graph with only feasible edges
        original_graph: The original complete graph for coordinate lookup
    
    Returns:
        str: Space-separated string of snapped waypoint names, or None if failed
    """
    if not original_route_str or not isinstance(original_route_str, str):
        return None
    
    waypoint_names = original_route_str.split()
    
    # Convert waypoint names to coordinates
    obs_pts = []
    for waypoint_name in waypoint_names:
        if waypoint_name in original_graph.nodes:
            node_data = original_graph.nodes[waypoint_name]
            obs_pts.append((node_data["lat"], node_data["lon"]))
        else:
            logging.warning(f"Waypoint '{waypoint_name}' not found in original graph")
    
    # Need at least 2 points for a route
    if len(obs_pts) < 2:
        logging.warning(f"Not enough waypoints found in graph ({len(obs_pts)})")
        return None
    
    # Check if feasible graph has any edges
    if feasible_graph.number_of_edges() == 0:
        logging.warning("Feasible graph has no edges")
        return None
    
    # Add length_nm attribute to feasible graph edges (required for viterbi_match)
    for u, v in feasible_graph.edges():
        if "length_nm" not in feasible_graph.edges[u, v]:
            lat1, lon1 = feasible_graph.nodes[u]["lat"], feasible_graph.nodes[u]["lon"]
            lat2, lon2 = feasible_graph.nodes[v]["lat"], feasible_graph.nodes[v]["lon"]
            feasible_graph.edges[u, v]["length_nm"] = haversine_nm(lat1, lon1, lat2, lon2)
    
    # Use viterbi_match to find the best path
    try:
        best_nodes, best_edges = viterbi_match(feasible_graph, obs_pts, k=50, beta=0.5)
        if not best_edges:
            logging.warning("No edges found in snapped route")
            return None
        
        # Construct the full node sequence from the edges
        full_nodes = [best_edges[0][0]] + [edge[1] for edge in best_edges]
        snapped_route_str = " ".join(full_nodes)
        return snapped_route_str
    
    except Exception as e:
        logging.error(f"Viterbi matching failed: {e}")
        return None


def run_tres_batch_processing(routes_csv_path, case_name, batch_size, config_path, output_dir_base, num_workers):
    """
    Main function to run the batch processing pipeline for TRES passes using multiprocessing.

    Args:
        routes_csv_path (str): Path to the routes CSV file.
        case_name (str): A name for this processing run, e.g., 'LEMD_EGLL'.
        batch_size (int): Number of flights per batch.
        config_path (str): Path to the base YAML configuration file.
        output_dir_base (str): The root directory where outputs will be saved.
        num_workers (int): The number of worker processes to use.
    """
    logging.info("Starting TRES batch processing pipeline with multiprocessing.")

    # 1. Load base configuration and initialize shared components
    logging.info(f"Loading base configuration from {config_path}")
    base_config = RunConfiguration.load_from_yaml(config_path)
    components = base_config.initialize_all_components()
    logging.info(f"Using device: {components['device']}")

    # 2. Get flight batches
    logging.info(f"Dividing flights from {routes_csv_path} into batches of size {batch_size}")
    flight_batches = get_flight_batches(routes_csv_path, batch_size)
    logging.info(f"Created {len(flight_batches)} batches.")
    
    # 3. Initialize CSV file for snapped routes
    snapped_routes_csv_path = os.path.join(output_dir_base, "all_routes_feasibly_snapped.csv")
    csv_fieldnames = [
        "flight_id", "route", "takeoff_time", "landing_time", 
        "cruise_altitude", "origin", "destination", "flight_time_s"
    ]
    initialize_csv_file(snapped_routes_csv_path, csv_fieldnames)
    logging.info(f"Initialized CSV file for snapped routes: {snapped_routes_csv_path}")

    # 4. Process each batch
    for i, batch_df in enumerate(flight_batches):
        batch_name = f"batch{i}"
        logging.info(f"--- Processing {batch_name} ({len(batch_df)} flights) ---")

        # Create batch output directory and save batch metadata
        batch_output_dir = os.path.join(output_dir_base, batch_name)
        os.makedirs(batch_output_dir, exist_ok=True)
        
        # Save the batch DataFrame as flights.csv
        flights_csv_path = os.path.join(batch_output_dir, "flights.csv")
        batch_df.to_csv(flights_csv_path, index=False)
        logging.info(f"Saved batch flights metadata to {flights_csv_path}")

        # Prepare a partial function with fixed arguments for the process pool
        worker_func = partial(
            process_flight,
            config=base_config,
            components=components,
            case_name=case_name,
            batch_idx=i,
            output_dir_base=output_dir_base,
            csv_path=snapped_routes_csv_path,
            csv_fieldnames=csv_fieldnames
        )

        # Create a list of flight_series objects for the workers
        flight_series_list = [row for _, row in batch_df.iterrows()]
        
        # Use a multiprocessing Pool to process flights in parallel
        # The 'fork' start method is generally faster but is not available on Windows.
        # 'spawn' is the default on Windows and macOS.
        # It's good practice to set it explicitly if you need cross-platform consistency.
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            # The list(...) and tqdm(...) pattern ensures we wait for all jobs to finish
            # and get a progress bar.
            list(tqdm(pool.imap_unordered(worker_func, flight_series_list), total=len(flight_series_list), desc=f"Flights in {batch_name}"))
                    
    # 5. Log completion
    logging.info(f"Snapped routes are being written on-the-fly to: {snapped_routes_csv_path}")

    logging.info("TRES batch processing pipeline finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run TRES passes for a batch of flights using multiprocessing.")
    # parser.add_argument("--routes-csv", required=False, default="D:\\project-equinox\\data\\cases\\LEMD_EGLL\\all_routes_sculpted.csv", help="Path to the flight routes CSV file.")
    parser.add_argument("--routes-csv", required=False, default="/Volumes/CrucialX/project-equinox/data/cases/LEMD_EGLL/all_routes_sculpted.csv", help="Path to the flight routes CSV file.")
    parser.add_argument("--case-name", required=False, default="LEMD_EGLL", help="A unique name for the city pair case (e.g., LEMD_EGLL).")
    parser.add_argument("--batch-size", type=int, default=25, help="Number of flights per processing batch.")
    # parser.add_argument("--config", required=False, default="D:\\project-equinox\\data\\cases\\LEMD_EGLL\\default.yaml", help="Path to the base YAML configuration file.")
    parser.add_argument("--config", required=False, default="/Volumes/CrucialX/project-equinox/data/cases/LEMD_EGLL/default.yaml", help="Path to the base YAML configuration file.")
    # parser.add_argument("--output-dir", default="D:\\project-equinox\\data\\cases\\LEMD_EGLL\\tres_runs", help="Base directory to save the output files.")
    parser.add_argument("--output-dir", default="/Volumes/CrucialX/project-equinox/data/cases/LEMD_EGLL/tres_runs", help="Base directory to save the output files.")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() - 1, help="Number of worker processes to use.")
    
    args = parser.parse_args()

    # It's crucial to protect the main execution block when using multiprocessing
    # on platforms that use 'spawn' or 'forkserver' (like Windows or macOS).
    run_tres_batch_processing(
        routes_csv_path=args.routes_csv,
        case_name=args.case_name,
        batch_size=args.batch_size,
        config_path=args.config,
        output_dir_base=args.output_dir,
        num_workers=args.num_workers
    )

    # Example usage from the command line:
    # python src/equinox/training/tres_batch.py \
    #   --routes-csv "data/routes/my_routes.csv" \
    #   --case-name "LEMD_EGLL_2023" \
    #   --batch-size 50 \
    #   --config "data/profiles/nbjet_35450_egll_lemd_2023_04_01.yaml" \
    #   --output-dir "data/tres_runs_output" \
    #   --num-workers 4
