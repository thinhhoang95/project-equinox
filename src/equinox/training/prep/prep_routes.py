from datetime import datetime
import pandas as pd
import os
from pathlib import Path
from typing import Optional
from tqdm import tqdm

def filter_routes_by_origin_dest(
    input_dir: str,
    origin: str,
    dest: str,
    output_dir: str,
    case_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Filter flights from CSV files by origin and destination airports.
    
    Args:
        input_dir: Directory containing CSV files with flight data
        origin: Origin airport code (e.g., 'LEMD')
        dest: Destination airport code (e.g., 'EGLL')
        output_dir: Base output directory
        case_name: Optional case name, defaults to f'{origin}_{dest}'
    
    Returns:
        DataFrame with filtered flights including takeoff and landing times
    """
    if case_name is None:
        case_name = f'{origin}_{dest}'
    
    # Create output directory
    output_path = Path(output_dir) / case_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_filtered_flights = []
    
    # Process all CSV files in input directory
    input_path = Path(input_dir)
    csv_files = list(input_path.glob('*.csv'))

    # Remove ._ files
    csv_files = [f for f in csv_files if not f.name.startswith('.')]
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return pd.DataFrame()
    
    print(f"Processing {len(csv_files)} CSV files...")
    
    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            if df.empty:
                continue
            
            # Extract origin and destination using vectorized operations
            waypoints_split = df['real_waypoints'].str.split()
            pass_times_split = df['pass_times'].str.split()
            
            # Add origin and destination columns
            df['origin'] = waypoints_split.str[0]
            df['destination'] = waypoints_split.str[-1]
            
            # Add takeoff and landing times
            df['takeoff'] = pass_times_split.str[0].astype(int)
            df['landing'] = pass_times_split.str[-1].astype(int)
            
            # Calculate flight time in seconds
            df['flight_time_s'] = df['landing'] - df['takeoff']
            
            # Filter using vectorized boolean indexing
            mask = (df['origin'] == origin) & (df['destination'] == dest)
            filtered_df = df[mask].copy()
            
            if not filtered_df.empty:
                all_filtered_flights.append(filtered_df)
                print(f"Found {len(filtered_df)} matching flights in {csv_file.name}")
        
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
            continue
    
    # Combine all filtered flights
    if all_filtered_flights:
        combined_df = pd.concat(all_filtered_flights, ignore_index=True)

        # Filter only for flights with takeoff in April 2023
        april_2023_timestamp = datetime(2023, 5, 1, 0, 0, 0).timestamp()
        combined_df = combined_df[combined_df['takeoff'] < april_2023_timestamp]
        
        # Save to output file
        output_file = output_path / 'all_routes.csv'
        combined_df.to_csv(output_file, index=False)
        
        print(f"Saved {len(combined_df)} filtered flights to {output_file}")
        return combined_df
    else:
        print(f"No flights found from {origin} to {dest}")
        return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    # Example: filter flights from Madrid to London Heathrow
    input_directory = "D:\\project-akrav\\matched_filtered_data"
    origin_airport = "LEMD"
    destination_airport = "EGLL"
    output_directory = "D:\\project-equinox\\data\\cases"
    
    filtered_flights = filter_routes_by_origin_dest(
        input_dir=input_directory,
        origin=origin_airport,
        dest=destination_airport,
        output_dir=output_directory
    )
    
    print(f"Filtered {len(filtered_flights)} flights")
