# `vnav_performance` Module Documentation

## `Performance` Class

Models aircraft performance characteristics based on speed and vertical speed profiles. It handles climb, cruise, and descent phases of flight.

### Initialization (`__init__`)

Initializes the `Performance` model.

**Arguments:**

-   `climb_speed_profile`: `List[Tuple[float, float]]` - TAS profile for climb `[(alt_thresh_ft, speed_kts), ...]`. Example: `[(10000, 250), (float('inf'), 300)]`
-   `descent_speed_profile`: `List[Tuple[float, float]]` - TAS profile for descent `[(alt_thresh_ft, speed_kts), ...]`. Example: `[(5000, 200), (float('inf'), 250)]`
-   `climb_vertical_speed_profile`: `List[Tuple[float, float]]` - VS profile for climb `[(alt_thresh_ft, vs_fpm), ...]`. VS values should be positive. Example: `[(5000, 1500), (float('inf'), 1000)]`
-   `descent_vertical_speed_profile`: `List[Tuple[float, float]]` - VS profile for descent `[(alt_thresh_ft, vs_fpm), ...]`. VS values should be negative for descent. Example: `[(10000, -1200), (float('inf'), -800)]`
-   `cruise_altitude_ft`: `float` - The designated cruise altitude in feet. Example: `35000`
-   `cruise_speed_kts`: `float` - The designated cruise true airspeed in knots. Example: `450`

**Raises:**

-   `ValueError`: If profiles are empty or not correctly terminated with `float('inf')`.

### Methods

#### `get_tas(self, altitude_ft: float, phase: str) -> float`

Returns the True Airspeed (TAS) in knots for a given altitude and flight phase.

**Arguments:**

-   `altitude_ft`: `float` - The altitude in feet.
-   `phase`: `str` - The flight phase. Must be `'climb'`, `'descent'`, or `'cruise'`.

**Returns:**

-   `float` - The True Airspeed (TAS) in knots at the given altitude for the specified phase.

#### `get_vertical_speed(self, altitude_ft: float, phase: str) -> float`

Returns the vertical speed (VS) in feet per minute (fpm) for a given altitude and flight phase. Positive for climb, negative for descent (assuming input profile provides this), zero for cruise.

**Arguments:**

-   `altitude_ft`: `float` - The altitude in feet.
-   `phase`: `str` - The flight phase. Must be `'climb'`, `'descent'`, or `'cruise'`.

**Returns:**

-   `float` - The vertical speed (VS) in feet per minute (fpm) at the given altitude for the specified phase.

#### `get_climb_eta(self, origin_airport_elevation_ft: float) -> List[Tuple[float, float]]`

Calculates the elapsed time (in seconds) since takeoff to reach each significant altitude level during climb, up to cruise altitude. A "significant altitude" is an altitude at which there is a change in target true airspeed or vertical speed, as defined in the climb profiles, or the cruise altitude itself.

**Arguments:**

-   `origin_airport_elevation_ft`: `float` - The elevation of the origin airport in feet.

**Returns:**

-   `List[Tuple[float, float]]` - A list of tuples `(altitude_ft, elapsed_time_seconds)`. The first entry is always `(origin_airport_elevation_ft, 0.0)`. Subsequent entries represent significant altitudes reached and the time taken.
    Example: `[(0.0, 0.0), (10000.0, 218.18), (28000.0, 818.18), (35000.0, 1168.18)]` (altitude in feet, time in seconds).

#### `get_descent_eta(self, destination_airport_elevation_ft: float) -> List[Tuple[float, float]]`

Calculates the elapsed time (in seconds) from Top of Descent (TOD) to reach each significant altitude level during descent, down to destination airport elevation. Descent starts from the aircraft's cruise altitude. A "significant altitude" is an altitude at which there is a change in target true airspeed or vertical speed, as defined in the descent profiles, or the destination airport elevation itself.

**Arguments:**

-   `destination_airport_elevation_ft`: `float` - The elevation of the destination airport in feet.

**Returns:**

-   `List[Tuple[float, float]]` - A list of tuples `(altitude_ft, elapsed_time_seconds_from_TOD)`. The first entry is always `(cruise_altitude_ft, 0.0)`. Subsequent entries represent significant altitudes reached and the time taken from TOD.
    Example: `[(35000.0, 0.0), (28000.0, 400.0), (10000.0, 1000.0), (0.0, 1380.0)]` (altitude in feet, time in seconds from TOD).

## Example Usage

```python
from equinox.vnav.vnav_performance import Performance
from equinox.vnav.vnav_profiles_rev1 import (
    NARROW_BODY_JET_CLIMB_PROFILE,
    NARROW_BODY_JET_DESCENT_PROFILE,
    NARROW_BODY_JET_CLIMB_VS_PROFILE,
    NARROW_BODY_JET_DESCENT_VS_PROFILE,
)

# Define performance profiles and cruise parameters
climb_speed = NARROW_BODY_JET_CLIMB_PROFILE
descent_speed = NARROW_BODY_JET_DESCENT_PROFILE
climb_vs = NARROW_BODY_JET_CLIMB_VS_PROFILE
descent_vs = NARROW_BODY_JET_DESCENT_VS_PROFILE
cruise_alt = 35000
cruise_spd = 450

# Create a Performance instance
flight_performance = Performance(
    climb_speed_profile=climb_speed,
    descent_speed_profile=descent_speed,
    climb_vertical_speed_profile=climb_vs,
    descent_vertical_speed_profile=descent_vs,
    cruise_altitude_ft=cruise_alt,
    cruise_speed_kts=cruise_spd,
)

# Get Climb ETA
origin_elevation = 0
climb_etas = flight_performance.get_climb_eta(
    origin_airport_elevation_ft=origin_elevation
)
print(f"Climb ETAs (Alt, Time in seconds): {climb_etas}")
# Expected output format: [(0.0, 0.0), (10000.0, ...), (28000.0, ...), (35000.0, ...)]

# Get Descent ETA
destination_elevation = 0
descent_etas = flight_performance.get_descent_eta(
    destination_airport_elevation_ft=destination_elevation
)
print(f"Descent ETAs (Alt, Time from TOD in seconds): {descent_etas}")
# Expected output format: [(35000.0, 0.0), (28000.0, ...), (10000.0, ...), (0.0, ...)]

# Get TAS and Vertical Speed at a specific altitude and phase
alt_query = 20000
phase_query = "climb"
tas_at_alt = flight_performance.get_tas(alt_query, phase_query)
vs_at_alt = flight_performance.get_vertical_speed(alt_query, phase_query)
print(f"At {alt_query} ft during {phase_query}: TAS = {tas_at_alt} kts, VS = {vs_at_alt} fpm")

phase_query = "descent"
tas_at_alt = flight_performance.get_tas(alt_query, phase_query)
vs_at_alt = flight_performance.get_vertical_speed(alt_query, phase_query)
print(f"At {alt_query} ft during {phase_query}: TAS = {tas_at_alt} kts, VS = {vs_at_alt} fpm")

phase_query = "cruise"
alt_query = 35000
tas_at_alt = flight_performance.get_tas(alt_query, phase_query)
vs_at_alt = flight_performance.get_vertical_speed(alt_query, phase_query)
print(f"At {alt_query} ft during {phase_query}: TAS = {tas_at_alt} kts, VS = {vs_at_alt} fpm")
```
