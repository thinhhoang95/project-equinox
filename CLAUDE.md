# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Equinox is a Python library for large-scale dynamic programming and reinforcement learning focused on air traffic control routing and weather-aware flight planning. It implements Maximum Entropy Inverse Learning to learn from historical flight data based on economic decision theory.

## Architecture

### Core Structure
- **src/equinox/**: Main package containing all modules
- **data/**: Flight cases, weather data (ERA5), graphs, and empirical data
- **notebooks/**: Jupyter notebooks for analysis and visualization
- **tests/**: Test files for various components

### Key Modules
- **config.py**: Central configuration management with `RunConfiguration` class
- **cost/**: Cost models (rev1-rev4) for flight path optimization
- **dp/**: Dynamic programming algorithms (forward/backward value iteration)
- **route/**: Flight state management and routing algorithms
- **wind/**: Weather model integration (ERA5 data processing)
- **vnav/**: Vertical navigation performance models
- **training/**: Batch SGD pipeline for model training
- **sampling/**: Route sampling and shortest path algorithms

### Data Flow
1. Graph preparation (`training/prep/`) - processes airspace graphs
2. Weather data processing (`wind/`) - handles ERA5 meteorological data
3. Dynamic programming (`dp/`) - computes optimal flight paths
4. Cost function learning (`cost/`) - learns from historical preferences
5. Training pipeline (`training/`) - batch processing with SGD optimization

## Development Commands

### Environment Setup
```bash
pip install -r requirements.txt
pip install -e .
```

### Testing
```bash
python -m pytest tests/
```

### Development Tools
- **Code formatting**: Uses `black` (configured in pyproject.toml)
- **Linting**: Uses `flake8` 
- **Type checking**: Uses `mypy`
- **Documentation**: Uses `sphinx` with RTD theme

### Running Development Tools
```bash
# Format code
black src/

# Lint code  
flake8 src/

# Type check
mypy src/
```

## Configuration Management

The project uses YAML-based configuration through the `RunConfiguration` class in `config.py`. Key configuration areas:

- **Graph paths**: Route network files and distance matrices
- **Weather data**: ERA5 data directory and date ranges
- **Cost models**: Parameters for different cost function versions (rev2-rev4)
- **Aircraft performance**: Speed profiles and altitude constraints
- **Training parameters**: SGD settings and regularization

Default configurations are stored in `data/profiles/` and can be loaded/saved via YAML.

## Key Algorithms

### Dynamic Programming
- **Forward DP**: Computes optimal paths from origin (`dp/forward_dp_vec*.py`)
- **Backward DP**: Computes value functions from destination (`dp/backward_dp_*.py`)
- **Soft Value Iteration**: Probabilistic path planning with temperature parameter

### Cost Learning
- **Maximum Entropy Inverse Learning**: Learns cost functions from observed flight preferences
- **Regularization**: Preference regularization to avoid overfitting
- **Multiple cost model versions**: Progressive improvements (rev1 through rev4)

### Training Pipeline
- **Batch processing**: Handles large-scale flight data
- **SGD optimization**: Stochastic gradient descent with TensorBoard logging
- **Checkpointing**: Model state saving/loading for long training runs

## Working with Flight Data

### Data Structure
- **Cases**: Each flight case contains routes, weather, and preference data
- **Graphs**: NetworkX format with waypoint coordinates and connections
- **Weather**: ERA5 NetCDF files with wind/temperature data
- **Trajectories**: Sampled flight paths with state information

### Common Workflows
1. **Preparing new flight case**: Use `training/prep/prep_graph.py` and `prep_routes.py`
2. **Training cost models**: Run `training/batch_sgd_pipeline.py` with appropriate config
3. **Evaluating results**: Use notebooks in `notebooks/` for visualization and analysis

## GPU/CPU Configuration

The project supports both CPU and GPU execution via PyTorch. Device selection is configured in `RunConfiguration.device_preference` and automatically falls back to CPU if CUDA is unavailable.

## Important File Patterns

- **`.pt` files**: PyTorch tensors (models, cached computations)
- **`.pkl` files**: Pickled Python objects (flight states, transitions)
- **`.gml` files**: NetworkX graph files
- **`.yaml` files**: Configuration files
- **`.nc` files**: NetCDF weather data files

## Branch Strategy

- **main**: Primary development branch for pull requests
- **batch_learning**: Current working branch for batch learning features