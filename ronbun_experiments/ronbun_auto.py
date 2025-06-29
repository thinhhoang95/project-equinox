#!/usr/bin/env python3
"""
Automated Testing Script for Route Sampling
=============================================

This script processes historical flights from all_routes.sculpted.csv and generates
sampled routes for each flight using the dynamic programming pipeline.

Features:
- Processes all flights in the specified case
- Creates individual YAML configurations from template
- Runs the sampling pipeline for each flight
- Organizes outputs in systematic directory structure
- Handles errors gracefully and continues processing
"""

import os
import sys
import csv
import yaml
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Global configuration
CASE_NAME = "LGAV_LFPG"
BASE_DATA_DIR = project_root / "data" / "cases"
RONBUN_DIR = project_root / "ronbun_experiments"
TEMPLATE_YAML = RONBUN_DIR / "template.yaml"

class FlightProcessor:
    """Handles processing of individual flights from CSV data."""
    
    def __init__(self, case_name: str = CASE_NAME):
        self.case_name = case_name
        self.case_dir = BASE_DATA_DIR / case_name
        self.csv_file = self.case_dir / "all_routes_sculpted.csv"
        self.template_yaml = TEMPLATE_YAML
        
        # Create organized directory structure
        self.setup_directories()
        self.setup_logging()
        
    def setup_directories(self):
        """Create systematic directory structure for outputs."""
        self.work_dir = RONBUN_DIR / f"runs_{self.case_name}"
        self.configs_dir = self.work_dir / "configs"
        self.samples_dir = self.work_dir / "samples"
        self.logs_dir = self.work_dir / "logs"
        self.trajectories_dir = self.work_dir / "trajectories"
        
        # Create all directories
        for dir_path in [self.work_dir, self.configs_dir, self.samples_dir, 
                        self.logs_dir, self.trajectories_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print(f"Working directory: {self.work_dir}")
        
    def setup_logging(self):
        """Setup logging for the automation process."""
        log_file = self.logs_dir / f"ronbun_auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_template_config(self) -> Dict:
        """Load the template YAML configuration."""
        if not self.template_yaml.exists():
            raise FileNotFoundError(f"Template YAML not found: {self.template_yaml}")
            
        with open(self.template_yaml, 'r') as f:
            return yaml.safe_load(f)
            
    def load_flights_data(self) -> pd.DataFrame:
        """Load and validate the flights CSV data."""
        if not self.csv_file.exists():
            raise FileNotFoundError(f"Flights CSV not found: {self.csv_file}")
            
        try:
            df = pd.read_csv(self.csv_file)
            required_columns = ['flight_id', 'route', 'takeoff_time', 'landing_time', 
                              'cruise_altitude', 'origin', 'destination']
            
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            self.logger.info(f"Loaded {len(df)} flights from {self.csv_file}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load flights data: {e}")
            raise
            
    def extract_route_waypoints(self, route_str: str) -> List[str]:
        """Extract waypoints from route string."""
        if pd.isna(route_str) or not route_str.strip():
            return []
        return [wp.strip() for wp in route_str.split() if wp.strip()]
        
    def unix_to_datetime_str(self, unix_timestamp: int) -> str:
        """Convert Unix timestamp to datetime string."""
        return datetime.fromtimestamp(unix_timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
    def create_unique_flight_id(self, flight_row: pd.Series) -> str:
        """Create unique flight ID by combining flight_id with takeoff_time."""
        flight_id = flight_row['flight_id']
        takeoff_time = int(flight_row['takeoff_time'])
        return f"{flight_id}_{takeoff_time}"
    
    def create_flight_config(self, flight_row: pd.Series, template_config: Dict) -> Dict:
        """Create flight-specific configuration from template."""
        config = template_config.copy()
        
        # Extract flight information
        original_flight_id = flight_row['flight_id']
        unique_flight_id = self.create_unique_flight_id(flight_row)
        route_waypoints = self.extract_route_waypoints(flight_row['route'])
        takeoff_time = self.unix_to_datetime_str(int(flight_row['takeoff_time']))
        landing_time = self.unix_to_datetime_str(int(flight_row['landing_time']))
        
        # Extract date for ERA5 wind data (use takeoff date)
        takeoff_datetime = datetime.fromtimestamp(int(flight_row['takeoff_time']))
        wind_date = takeoff_datetime.strftime('%Y-%m-%d')
        
        # Update configuration with flight-specific data
        config.update({
            'file_prefix': unique_flight_id,
            'origin_node': flight_row['origin'],
            'goal_node': flight_row['destination'],
            'takeoff_time_str': takeoff_time,
            'landing_time_str': landing_time,
            'estimated_takeoff_time_str': takeoff_time,
            'estimated_landing_time_str': landing_time,
            'cruise_altitude_ft': float(flight_row['cruise_altitude']) * 3.28084,
            'output_dir': str(self.samples_dir / unique_flight_id),
            'wind_date': wind_date
        })
        
        # Update paths to point to the correct case
        case_graph_dir = self.case_dir / "graphs"
        config.update({
            'graph_file_path': str(case_graph_dir / "routes.gml"),
            'distances_file_path': str(case_graph_dir / "routes_distances.npy"),
            'charges_file_path': str(case_graph_dir / "routes_charges.npy"),
        })
        
        return config
        
    def save_flight_config(self, unique_flight_id: str, config: Dict) -> Path:
        """Save flight configuration to YAML file."""
        config_file = self.configs_dir / f"{unique_flight_id}.yaml"
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
        return config_file
        
    def run_sampling_pipeline(self, config_file: Path, flight_id: str) -> bool:
        """Run the sampling pipeline for a single flight."""
        try:
            self.logger.info(f"Starting pipeline for flight {flight_id}")
            
            # Import the modified sample1 functions
            sys.path.insert(0, str(RONBUN_DIR))
            from sample1_modified import run_full_pipeline
            
            # Run the pipeline
            success = run_full_pipeline(str(config_file))
            
            if success:
                self.logger.info(f"Pipeline completed successfully for flight {flight_id}")
                return True
            else:
                self.logger.error(f"Pipeline failed for flight {flight_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error running pipeline for flight {flight_id}: {e}")
            return False
            
    def extract_sampled_routes(self, unique_flight_id: str) -> bool:
        """Extract and save sampled routes to text file."""
        try:
            # Look for trajectory file
            flight_samples_dir = self.samples_dir / unique_flight_id
            trajectory_file = flight_samples_dir / f"{unique_flight_id}_CLB_trajectories.txt"
            
            if not trajectory_file.exists():
                self.logger.warning(f"Trajectory file not found for flight {unique_flight_id}")
                return False
                
            # Copy trajectory file to organized location
            output_file = self.trajectories_dir / f"{unique_flight_id}.txt"
            
            # Read and process trajectory file
            with open(trajectory_file, 'r') as f_in, open(output_file, 'w') as f_out:
                for line in f_in:
                    # Each line format: "cost,waypoint1 waypoint2 ..."
                    f_out.write(line)
                    
            self.logger.info(f"Saved sampled routes for flight {unique_flight_id} to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error extracting routes for flight {unique_flight_id}: {e}")
            return False
            
    def process_single_flight(self, flight_row: pd.Series, template_config: Dict) -> bool:
        """Process a single flight through the complete pipeline."""
        original_flight_id = flight_row['flight_id']
        unique_flight_id = self.create_unique_flight_id(flight_row)
        
        try:
            # Create flight-specific configuration
            flight_config = self.create_flight_config(flight_row, template_config)
            
            # Save configuration file
            config_file = self.save_flight_config(unique_flight_id, flight_config)
            self.logger.info(f"Created config for flight {original_flight_id} (unique: {unique_flight_id}): {config_file}")
            
            # Run sampling pipeline
            if not self.run_sampling_pipeline(config_file, unique_flight_id):
                return False
                
            # Extract and organize results
            if not self.extract_sampled_routes(unique_flight_id):
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process flight {original_flight_id} (unique: {unique_flight_id}): {e}")
            return False
            
    def run_batch_processing(self, max_flights: Optional[int] = None, 
                           start_from: int = 0) -> Tuple[int, int]:
        """Run batch processing of all flights."""
        try:
            # Load template and flights data
            template_config = self.load_template_config()
            flights_df = self.load_flights_data()
            
            # Apply limits if specified
            if start_from > 0:
                flights_df = flights_df.iloc[start_from:]
            if max_flights:
                flights_df = flights_df.head(max_flights)
                
            total_flights = len(flights_df)
            successful = 0
            failed = 0
            
            self.logger.info(f"Starting batch processing of {total_flights} flights")
            
            # Process each flight
            for idx, (_, flight_row) in enumerate(flights_df.iterrows(), 1):
                original_flight_id = flight_row['flight_id']
                unique_flight_id = self.create_unique_flight_id(flight_row)

                # Check if trajectory file already exists
                trajectory_file = self.trajectories_dir / f"{unique_flight_id}.txt"
                if trajectory_file.exists():
                    self.logger.info(f"Skipping flight {original_flight_id} (unique: {unique_flight_id}) ({idx}/{total_flights}): Trajectory file already exists.")
                    successful += 1
                    continue
                
                self.logger.info(f"Processing flight {idx}/{total_flights}: {original_flight_id} (unique: {unique_flight_id})")
                
                start_time = time.time()
                success = self.process_single_flight(flight_row, template_config)
                elapsed = time.time() - start_time
                
                if success:
                    successful += 1
                    self.logger.info(f" Flight {original_flight_id} (unique: {unique_flight_id}) completed in {elapsed:.1f}s")
                else:
                    failed += 1
                    self.logger.error(f" Flight {original_flight_id} (unique: {unique_flight_id}) failed after {elapsed:.1f}s")
                    
                # Progress update
                if idx % 10 == 0:
                    self.logger.info(f"Progress: {idx}/{total_flights} flights processed "
                                   f"({successful} successful, {failed} failed)")
                    
            self.logger.info(f"Batch processing completed: {successful} successful, {failed} failed")
            return successful, failed
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            raise
            
    def generate_summary_report(self, successful: int, failed: int):
        """Generate a summary report of the processing."""
        report_file = self.work_dir / "processing_summary.txt"
        
        with open(report_file, 'w') as f:
            f.write(f"Ronbun Automated Processing Summary\n")
            f.write(f"==================================\n\n")
            f.write(f"Case: {self.case_name}\n")
            f.write(f"Processed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Results:\n")
            f.write(f"- Successful flights: {successful}\n")
            f.write(f"- Failed flights: {failed}\n")
            f.write(f"- Total flights: {successful + failed}\n")
            f.write(f"- Success rate: {successful/(successful+failed)*100:.1f}%\n\n")
            f.write(f"Output directories:\n")
            f.write(f"- Configurations: {self.configs_dir}\n")
            f.write(f"- Samples: {self.samples_dir}\n")
            f.write(f"- Trajectories: {self.trajectories_dir}\n")
            f.write(f"- Logs: {self.logs_dir}\n")
            
        self.logger.info(f"Summary report saved to: {report_file}")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated flight route sampling")
    parser.add_argument("--case", default=CASE_NAME, help="Case name to process")
    parser.add_argument("--max-flights", type=int, help="Maximum number of flights to process")
    parser.add_argument("--start-from", type=int, default=0, help="Start processing from flight index")
    parser.add_argument("--test", action="store_true", help="Run test with first 3 flights")
    
    args = parser.parse_args()
    
    if args.test:
        args.max_flights = 3
        print("Running in test mode with first 3 flights")
    
    try:
        # Initialize processor
        processor = FlightProcessor(args.case)
        
        # Run batch processing
        successful, failed = processor.run_batch_processing(
            max_flights=args.max_flights,
            start_from=args.start_from
        )
        
        # Generate summary report
        processor.generate_summary_report(successful, failed)
        
        print(f"\nProcessing completed: {successful} successful, {failed} failed")
        return 0 if failed == 0 else 1
        
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())