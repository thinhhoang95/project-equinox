import pandas as pd
import os
from datetime import datetime
from copy import deepcopy
import logging
from tqdm import tqdm
import argparse
import multiprocessing
from functools import partial

from equinox.wind.batch_wind_model import get_flight_batches
from equinox.wind.wind_date import WindDate
from equinox.config import RunConfiguration
from equinox.dp.trespass.tres_forward import tres_forward, save_transitions
from equinox.dp.trespass.tres_backward import tres_backward

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_flight(flight_series, config, components, case_name, batch_idx, output_dir_base):
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
    """
    flight_id = flight_series['flight_id']
    takeoff_ts = flight_series['takeoff']

    # This log might be helpful for debugging in a multiprocessing context
    # logging.info(f"Processing flight {flight_id} with takeoff {takeoff_ts}")

    # 1. Prepare flight-specific configuration and create a dedicated wind model
    flight_config = deepcopy(config)
    flight_config.origin_node = flight_series['origin']
    flight_config.goal_node = flight_series['destination']

    dt_takeoff = datetime.fromtimestamp(takeoff_ts)
    dt_landing = datetime.fromtimestamp(flight_series['landing'])

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

    # 4. Process each batch
    for i, batch_df in enumerate(flight_batches):
        batch_name = f"batch{i}"
        logging.info(f"--- Processing {batch_name} ({len(batch_df)} flights) ---")

        # Prepare a partial function with fixed arguments for the process pool
        worker_func = partial(
            process_flight,
            config=base_config,
            components=components,
            case_name=case_name,
            batch_idx=i,
            output_dir_base=output_dir_base
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
            results = list(tqdm(pool.imap_unordered(worker_func, flight_series_list), total=len(flight_series_list), desc=f"Flights in {batch_name}"))

    logging.info("TRES batch processing pipeline finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run TRES passes for a batch of flights using multiprocessing.")
    parser.add_argument("--routes-csv", required=False, default="D:\\project-equinox\\data\\cases\\LEMD_EGLL\\all_routes.csv", help="Path to the flight routes CSV file.")
    parser.add_argument("--case-name", required=False, default="LEMD_EGLL", help="A unique name for the city pair case (e.g., LEMD_EGLL).")
    parser.add_argument("--batch-size", type=int, default=25, help="Number of flights per processing batch.")
    parser.add_argument("--config", required=False, default="D:\\project-equinox\\data\\cases\\LEMD_EGLL\\default.yaml", help="Path to the base YAML configuration file.")
    parser.add_argument("--output-dir", default="D:\\project-equinox\\data\\cases\\LEMD_EGLL\\tres_runs", help="Base directory to save the output files.")
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
