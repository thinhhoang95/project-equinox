# Conductor.md - Equinox System Documentation

## Overview

This document provides comprehensive documentation of the Equinox system's architecture, logic, and interconnections between different functions. Equinox is a Python library implementing Maximum Entropy Inverse Learning for air traffic control routing and weather-aware flight planning using dynamic programming and reinforcement learning.

## System Architecture

### Core Design Principles

1. **Configuration-Driven Architecture**: All system components are initialized and configured through the central `RunConfiguration` class
2. **Modular Design**: Clear separation of concerns with distinct modules for different functionalities
3. **Device Agnostic**: Automatic CPU/GPU selection with fallback mechanisms
4. **Batch Processing**: Designed for large-scale processing of flight data
5. **Extensible Cost Models**: Progressive cost model versions (rev1-rev4) with backward compatibility

### High-Level Data Flow

```
Configuration (config.py) → Component Initialization → Data Processing → Training Pipeline → Model Optimization
```

## Module Architecture & Interconnections

### 1. Configuration Management (`config.py`)

**Purpose**: Central configuration hub that manages all system parameters and component initialization.

**Key Classes**:
- `RunConfiguration`: Dataclass containing all system parameters

**Key Functions**:
- `initialize_all_components()`: Creates and configures all system components
- `get_cost_model_class()`: Factory function for cost model selection
- `load_graph()`: Loads route network and creates node mappings
- `initialize_wind_model()`: Creates weather model instances
- `initialize_performance_model()`: Creates aircraft performance models

**Relationships**:
- **Imports**: All modules depend on config for initialization
- **Exports**: Configuration parameters to all components
- **Factory Pattern**: Creates instances of cost models, wind models, and performance models

### 2. Cost Models (`cost/`)

**Purpose**: Implements various cost functions for flight path optimization using maximum entropy inverse learning.

**Key Components**:

#### Cost Model Evolution
- **`CostRev1`**: Basic cost model with simple linear functions
- **`CostRev2`**: Regularized version with preference regularization (`cost_rev2_reg.py`)
- **`CostRev3`**: Monotonic piecewise linear functions (`cost_rev3.py`)
- **`CostRev4`**: Advanced piecewise linear models with airspace charges (`cost_rev4.py`)
- **`CostRev4Lite`**: Lightweight version of CostRev4 (`cost_rev4_lite.py`)
- **`CostRev4Ronbun1`**: Research variant for academic publications (`cost_rev4_ronbun1.py`)

#### Supporting Classes
- **`PiecewiseLinearMonoModel`**: Monotonic piecewise linear function implementation (`plf_mono.py`)
- **`PLF`**: General piecewise linear function utilities (`plf.py`)

**Key Functions**:
- `forward()`: Computes cost for given state transitions
- `get_route_cost()`: Calculates total route cost for evaluation
- `regularization_loss()`: Computes preference regularization penalties

**Relationships**:
- **Used by**: Dynamic programming modules (`dp/`), training pipeline (`training/`)
- **Depends on**: Configuration (`config.py`), feature engineering (`feateng/`)
- **Data Flow**: State transitions → Cost computation → Gradient computation → Parameter updates

### 3. Dynamic Programming (`dp/`)

**Purpose**: Implements forward and backward dynamic programming algorithms for optimal path planning.

**Key Components**:

#### Forward Dynamic Programming
- **`forward_dp_vec1.py`**: Vectorized forward value iteration
- **`forward_dp_vec2.py`**: Enhanced vectorized implementation
- **`forward_soft_bellman.py`**: Soft Bellman updates with temperature parameter

#### Backward Dynamic Programming
- **`backward_dp_vec1.py`**: Backward value iteration with basic gradient computation
- **`backward_dp_vec2.py`**: Enhanced backward DP with improved gradient handling
- **`backward_dp_opt.py`**: Optimized backward DP implementation

#### TRES (Trajectory Reachability) Module
- **`tres_forward.py`**: Forward trajectory reachability analysis
- **`tres_backward.py`**: Backward trajectory reachability analysis
- **`thinning.py`**: State space reduction through reachability analysis

#### Advanced Optimization (`trespass/amorwin/`)
- **`backward_gradient.py`**: Gradient computation for backward passes
- **`backward_svi_log_cost.py`**: Soft value iteration with logarithmic cost functions
- **`forward_svi_log.py`**: Forward soft value iteration with temperature

**Key Functions**:
- `forward_dp()`: Forward dynamic programming computation
- `backward_dp()`: Backward dynamic programming with gradient computation
- `soft_bellman_update()`: Probabilistic policy updates
- `compute_gradients()`: Gradient computation for learning

**Relationships**:
- **Used by**: Training pipeline (`training/`), route sampling (`sampling/`)
- **Depends on**: Cost models (`cost/`), route management (`route/`), configuration (`config.py`)
- **Data Flow**: State transitions → Value iteration → Optimal policies → Gradient computation

### 4. Route Management (`route/`)

**Purpose**: Manages flight states, transitions, and route computations with aircraft performance models.

**Key Components**:
- **`forward_state.py`**: Forward state propagation with performance constraints
- **`backward_state.py`**: Backward state propagation for gradient computation
- **`fms.py`**: Flight Management System functionality
- **`get_wind.py`**: Wind data interpolation for route planning
- **`value_iteration.py`**: Value iteration implementations
- **`batch_interpolator.py`**: Batch processing for wind interpolation

**Key Functions**:
- `forward_state_transition()`: Computes forward state transitions
- `backward_state_transition()`: Computes backward state transitions
- `get_wind_at_state()`: Interpolates wind data for specific states
- `compute_transition_cost()`: Calculates transition costs

**Relationships**:
- **Used by**: Dynamic programming (`dp/`), training pipeline (`training/`)
- **Depends on**: Wind models (`wind/`), performance models (`vnav/`), configuration (`config.py`)
- **Data Flow**: State definitions → Transition computation → Cost calculation → Policy updates

### 5. Wind Models (`wind/`)

**Purpose**: Integrates weather data (ERA5) into flight planning and provides wind interpolation.

**Key Components**:
- **`WindModel`**: Base class for weather data processing (`wind_model.py`)
- **`WindDate`**: Date-specific wind model using ERA5 data (`wind_date.py`)
- **`WindFree`**: Wind-free model for baseline comparisons (`wind_free.py`)
- **`batch_wind_model.py`**: Batch processing of wind data for multiple flights

**Key Functions**:
- `get_wind_at_position()`: Interpolates wind data at specific coordinates
- `preprocess_wind_data()`: Processes ERA5 NetCDF files
- `batch_wind_interpolation()`: Efficient batch wind interpolation

**Relationships**:
- **Used by**: Route management (`route/`), dynamic programming (`dp/`)
- **Depends on**: Configuration (`config.py`), helper functions (`helpers/`)
- **Data Flow**: ERA5 data → Wind interpolation → Route cost computation → Optimization

### 6. Vertical Navigation (`vnav/`)

**Purpose**: Provides aircraft performance models including climb/descent profiles and speed constraints.

**Key Components**:
- **`Performance`**: Main aircraft performance model (`vnav_performance.py`)
- **`vnav_profiles_rev1.py`**: Predefined aircraft performance profiles for different aircraft types

**Predefined Aircraft Profiles**:
- `NARROW_BODY_JET`: Standard commercial aircraft (A320, B737)
- `WIDE_BODY_JET`: Long-haul aircraft (A330, B777)
- `BUSINESS_JET`: Business aviation aircraft

**Key Functions**:
- `get_climb_speed()`: Returns climb speed for given altitude
- `get_descent_speed()`: Returns descent speed for given altitude
- `get_vertical_speed()`: Returns vertical speed profiles
- `compute_performance_constraints()`: Calculates performance-based constraints

**Relationships**:
- **Used by**: Route management (`route/`), dynamic programming (`dp/`)
- **Depends on**: Configuration (`config.py`)
- **Data Flow**: Performance profiles → State transitions → Route optimization

### 7. Training Pipeline (`training/`)

**Purpose**: Orchestrates large-scale training using batch SGD with comprehensive data preparation.

**Key Components**:

#### Main Training Scripts
- **`batch_sgd_pipeline.py`**: Main training orchestration
- **`batch_sgd_pipeline_parallel.py`**: Parallel processing version
- **`train.py`**: Configuration-based training interface

#### Data Preparation (`prep/`)
- **`prep_graph.py`**: Graph preprocessing and validation
- **`prep_routes.py`**: Route data preparation and matching
- **`resculpt_viterbi.py`**: Route matching using Viterbi algorithm

#### Batch Processing
- **`thin_batch.py`**: Batch thinning operations for state space reduction
- **`tres_batch.py`**: Batch TRES (trajectory reachability) computations
- **`svi_batch.py`**: Batch soft value iteration

**Key Functions**:
- `run_batch_sgd()`: Main training loop with SGD optimization
- `prepare_training_data()`: Prepares flight cases for training
- `compute_batch_gradients()`: Computes gradients for batch updates
- `save_checkpoint()`: Saves training state for resumption

**Relationships**:
- **Uses**: All core modules (dp/, cost/, route/, wind/, vnav/)
- **Depends on**: Configuration (`config.py`), sampling (`sampling/`)
- **Data Flow**: Flight cases → Batch processing → Gradient computation → Parameter updates

### 8. Sampling (`sampling/`)

**Purpose**: Provides route sampling and shortest path algorithms for evaluation and analysis.

**Key Components**:
- **`sample_route.py`**: Route sampling from learned policies
- **`shortest_path.py`**: Shortest path computations
- **`get_route_cost.py`**: Route cost evaluation utilities
- **`trespass/sampler.py`**: Advanced sampling with trajectory reachability

**Key Functions**:
- `sample_route_from_policy()`: Samples routes from learned policies
- `compute_shortest_path()`: Computes optimal routes
- `evaluate_route_quality()`: Evaluates sampled routes

**Relationships**:
- **Used by**: Training pipeline (`training/`), evaluation scripts
- **Depends on**: Dynamic programming (`dp/`), cost models (`cost/`)
- **Data Flow**: Learned policies → Route sampling → Evaluation → Analysis

### 9. Feature Engineering (`feateng/`)

**Purpose**: Provides specialized feature extraction for different cost components.

**Key Components**:
- **`distance.py`**: Distance calculations and features
- **`airspace_charges.py`**: Airspace charge computations
- **`laplace.py`**: Laplace approximation utilities

**Key Functions**:
- `compute_distance_features()`: Extracts distance-based features
- `compute_airspace_charges()`: Calculates airspace usage costs
- `apply_laplace_smoothing()`: Applies Laplace smoothing to features

**Relationships**:
- **Used by**: Cost models (`cost/`), route management (`route/`)
- **Depends on**: Helper functions (`helpers/`)
- **Data Flow**: Raw state data → Feature extraction → Cost computation

### 10. Helper Functions (`helpers/`)

**Purpose**: Provides common utilities used across the system.

**Key Components**:
- **`haversine.py`**: Great circle distance calculations
- **`datetimeh.py`**: Date/time utilities
- **`plotters.py`**: Visualization utilities

**Key Functions**:
- `haversine_distance()`: Calculates great circle distances
- `parse_flight_time()`: Parses flight time strings
- `plot_flight_trajectory()`: Visualizes flight paths

**Relationships**:
- **Used by**: All modules requiring common utilities
- **Standalone**: Minimal dependencies on other modules
- **Data Flow**: Utility functions → Supporting computations across modules

## Data Flow Diagrams

### 1. System Initialization Flow

```
RunConfiguration → load_graph() → Graph + Node Mappings
                → initialize_cost_model() → Cost Model Instance
                → initialize_wind_model() → Wind Model Instance
                → initialize_performance_model() → Performance Model Instance
                → initialize_all_components() → Complete System State
```

### 2. Training Pipeline Flow

```
Flight Cases → prep_graph.py → Preprocessed Graphs
            → prep_routes.py → Route Matching
            → batch_sgd_pipeline.py → Batch Processing
            → Dynamic Programming → Value Iteration
            → Cost Models → Gradient Computation
            → Parameter Updates → Model Optimization
```

### 3. Route Optimization Flow

```
Origin/Destination → Graph Loading → Route Network
                  → Wind Data → ERA5 Processing
                  → Performance Model → Aircraft Constraints
                  → Dynamic Programming → Optimal Policy
                  → Route Sampling → Evaluated Routes
```

## Algorithm Interconnections

### 1. Maximum Entropy Inverse Learning Pipeline

```
Historical Flight Data → Feature Extraction → Cost Model Training
                      → Preference Learning → Parameter Optimization
                      → Policy Evaluation → Route Generation
```

### 2. Dynamic Programming Integration

```
Forward DP: Origin → Intermediate States → Destination (Optimal Values)
Backward DP: Destination → Intermediate States → Origin (Gradients)
Integration: Forward Values + Backward Gradients → Parameter Updates
```

### 3. Multi-Model Training Process

```
Cost Model Rev1 → Rev2 (Regularization) → Rev3 (Monotonicity) → Rev4 (Piecewise Linear)
Each revision builds on previous versions with enhanced capabilities
```

## Key Design Patterns

### 1. Factory Pattern
- `get_cost_model_class()`: Creates cost model instances based on version
- `initialize_wind_model()`: Creates wind model instances based on configuration
- `initialize_performance_model()`: Creates performance model instances

### 2. Configuration Pattern
- `RunConfiguration`: Central configuration object
- YAML serialization/deserialization
- Component initialization through configuration

### 3. Batch Processing Pattern
- `batch_sgd_pipeline.py`: Main batch processing orchestration
- `thin_batch.py`: Batch state space reduction
- `tres_batch.py`: Batch trajectory reachability

### 4. State Pattern
- `forward_state.py`: Forward state transitions
- `backward_state.py`: Backward state transitions
- State-based route optimization

## Performance Considerations

### 1. GPU/CPU Optimization
- Automatic device selection in `RunConfiguration.get_device()`
- PyTorch tensor operations optimized for GPU acceleration
- Fallback to CPU when GPU unavailable

### 2. Memory Management
- Batch processing to handle large-scale data
- State space reduction through thinning algorithms
- Efficient tensor operations in PyTorch

### 3. Computational Efficiency
- Vectorized operations in dynamic programming
- Sparse matrix operations for large graphs
- Parallel processing in training pipeline

## Extension Points

### 1. Adding New Cost Models
1. Create new cost model class inheriting from appropriate base
2. Update `get_cost_model_class()` factory function
3. Add configuration parameters to `RunConfiguration`
4. Implement required methods: `forward()`, `regularization_loss()`

### 2. Adding New Aircraft Types
1. Define performance profiles in `vnav_profiles_rev1.py`
2. Add aircraft model key to configuration
3. Update `initialize_performance_model()` to handle new type

### 3. Adding New Wind Models
1. Create new wind model class inheriting from `WindModel`
2. Update `initialize_wind_model()` factory function
3. Implement required methods: `get_wind_at_position()`

## Testing Strategy

### 1. Unit Tests
- Individual module testing in `tests/` directory
- Mock dependencies for isolated testing
- PyTorch tensor operation validation

### 2. Integration Tests
- End-to-end pipeline testing
- Configuration validation
- Component initialization testing

### 3. Performance Tests
- Large-scale batch processing validation
- Memory usage monitoring
- GPU/CPU performance comparison

## Deployment Considerations

### 1. Environment Setup
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Data Requirements
- ERA5 weather data in NetCDF format
- Route network files in GML format
- Flight case data in structured format

### 3. Configuration Management
- YAML configuration files for different scenarios
- Environment-specific settings
- Checkpoint management for long-running training

## Future Development Directions

### 1. Enhanced Cost Models
- Integration of fuel consumption models
- Real-time weather adaptation
- Multi-objective optimization

### 2. Scalability Improvements
- Distributed training across multiple GPUs
- Advanced batch processing optimizations
- Real-time route optimization

### 3. Additional Features
- Real-time weather integration
- Advanced aircraft performance models
- Enhanced visualization capabilities

## Conclusion

The Equinox system represents a sophisticated implementation of maximum entropy inverse learning for air traffic control routing. Its modular architecture, comprehensive configuration management, and efficient algorithms make it suitable for large-scale flight planning optimization. The clear separation of concerns and extensible design enable future enhancements while maintaining system stability and performance.

This documentation serves as a comprehensive guide for developers working with the Equinox system, providing insights into the logical connections between components and the overall system architecture.