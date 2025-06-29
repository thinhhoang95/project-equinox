# Ronbun Automated Testing Framework

This directory contains the automated testing framework for processing historical flights and generating sampled routes.

## Files

- `ronbun_auto.py` - Main automation script
- `sample1_modified.py` - Modified version of sample1.py for batch processing
- `template.yaml` - Template configuration file
- `README.md` - This documentation file

## Quick Start

### 1. Test with Sample Flights
```bash
python ronbun_auto.py --test
```
This runs the automation on the first 3 flights as a test.

### 2. Process All Flights in Default Case (LGAV_LFPG)
```bash
python ronbun_auto.py
```

### 3. Process Specific Number of Flights
```bash
python ronbun_auto.py --max-flights 10
```

### 4. Process Different Case
```bash
python ronbun_auto.py --case LEMD_EGLL
```

### 5. Resume from Specific Flight Index
```bash
python ronbun_auto.py --start-from 50 --max-flights 20
```

## Configuration

### Global Variables
You can modify these at the top of `ronbun_auto.py`:
- `CASE_NAME`: Default case to process (default: "LGAV_LFPG")

### Template Configuration
The `template.yaml` file contains the base configuration that gets customized for each flight:
- Flight-specific data (takeoff/landing times, route endpoints, etc.) are automatically updated
- Graph file paths are automatically adjusted for the selected case
- Output directories are systematically organized

## Output Structure

When you run the automation, it creates a structured output directory:

```
ronbun_experiments/
├── runs_CASE_NAME/
│   ├── configs/          # Individual YAML configs for each flight
│   │   ├── FLIGHT_ID1.yaml
│   │   ├── FLIGHT_ID2.yaml
│   │   └── ...
│   ├── samples/          # Raw pipeline outputs for each flight
│   │   ├── FLIGHT_ID1/
│   │   ├── FLIGHT_ID2/
│   │   └── ...
│   ├── trajectories/     # Final trajectory files (one per flight)
│   │   ├── FLIGHT_ID1.txt
│   │   ├── FLIGHT_ID2.txt
│   │   └── ...
│   ├── logs/            # Processing logs
│   │   └── ronbun_auto_TIMESTAMP.log
│   └── processing_summary.txt  # Summary report
```

## Input Data Format

The script expects a CSV file named `all_routes_sculpted.csv` in the case directory with columns:
- `flight_id`: Unique identifier for the flight
- `route`: Space-separated waypoint names
- `takeoff_time`: Unix timestamp
- `landing_time`: Unix timestamp  
- `cruise_altitude`: Altitude in feet
- `origin`: Origin airport code
- `destination`: Destination airport code

## Pipeline Stages

For each flight, the automation runs these stages:
1. **Forward Tres**: Forward dynamic programming pass
2. **Backward Tres**: Backward dynamic programming pass
3. **Thinning**: State space reduction
4. **Wind Averaging**: Pre-compute wind effects
5. **Forward SVI**: Forward soft value iteration
6. **Backward SVI**: Backward soft value iteration  
7. **Trajectory Sampling**: Generate sampled routes

## Error Handling

- Individual flight failures don't stop the batch processing
- Detailed logs are saved for debugging
- Progress is reported every 10 flights
- Summary report shows success/failure statistics

## Requirements

- Python environment with equinox package installed
- PyTorch and other dependencies from requirements.txt
- Access to case data files (graphs, wind data, etc.)
- Sufficient disk space for intermediate files

## Customization

### Adding New Cases
1. Ensure the case directory exists in `data/cases/CASE_NAME/`
2. Verify required files are present:
   - `all_routes_sculpted.csv`
   - `graphs/routes.gml`
   - `graphs/routes_distances.npy`
   - `graphs/routes_charges.npy`

### Modifying Pipeline
Edit `sample1_modified.py` to change the processing pipeline or add new stages.

### Changing Configuration Template
Modify `template.yaml` to adjust default parameters for all flights.

## Troubleshooting

### Common Issues
1. **Import errors**: Ensure Python environment has all required packages
2. **File not found**: Check that case directories and files exist
3. **Memory issues**: Reduce batch size or process flights in smaller chunks
4. **GPU issues**: Set `device_preference: cpu` in template if CUDA unavailable

### Getting Help
Check the log files in the `logs/` directory for detailed error information.