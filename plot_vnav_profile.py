
import matplotlib.pyplot as plt
import numpy as np

# Assuming these profiles are available from src/equinox/vnav/vnav_profiles_rev1.py
# For demonstration, I'll hardcode them here. In a real scenario, you'd import them.
NARROW_BODY_JET_CLIMB_PROFILE = [
    (10000, 250),
    (20000, 330),
    (28000, 390),
    (float("inf"), 460),
]
NARROW_BODY_JET_DESCENT_PROFILE = [
    (10000, 250),
    (20000, 300),
    (28000, 360),
    (float("inf"), 450),
]
NARROW_BODY_JET_CLIMB_VS_PROFILE = [(10000, 3000), (28000, 2000), (float("inf"), 1200)]
NARROW_BODY_JET_DESCENT_VS_PROFILE = [
    (10000, 1500),
    (24000, 3000),
    (float("inf"), 1000),
]

def get_profile_value(profile, altitude, default_value):
    """Helper to get the appropriate value (speed or VS) from a profile based on altitude."""
    for alt_limit, value in profile:
        if altitude <= alt_limit:
            return value
    return default_value

def generate_flight_path(
    climb_profile,
    climb_vs_profile,
    descent_profile,
    descent_vs_profile,
    cruise_altitude=35000,
    takeoff_altitude=0,
    landing_altitude=0,
    horizontal_speed_kts=400, # knots
    time_step_seconds=60, # 1 minute
):
    """Generates flight path points (horizontal_distance, altitude) and speeds."""
    path = []
    speeds = []
    current_altitude = takeoff_altitude
    current_distance = 0

    # Takeoff to Cruise
    while current_altitude < cruise_altitude:
        path.append((current_distance, current_altitude))
        
        target_speed = get_profile_value(climb_profile, current_altitude, horizontal_speed_kts)
        vertical_speed_fpm = get_profile_value(climb_vs_profile, current_altitude, 1000) # feet per minute

        speeds.append(target_speed)

        # Convert vertical speed from fpm to feet per second
        vertical_speed_fps = vertical_speed_fpm / 60
        
        # Calculate altitude change
        altitude_change = vertical_speed_fps * time_step_seconds
        current_altitude += altitude_change

        # Calculate horizontal distance change (assuming constant horizontal speed)
        # Convert knots to feet per second (1 knot = 1.68781 feet/second)
        horizontal_speed_fps = horizontal_speed_kts * 1.68781
        distance_change = horizontal_speed_fps * time_step_seconds
        current_distance += distance_change

        if current_altitude >= cruise_altitude:
            current_altitude = cruise_altitude # Cap at cruise altitude
            path.append((current_distance, current_altitude))
            speeds.append(target_speed) # Add speed for the cruise entry point
            break

    # Cruise segment (simplified: fixed duration)
    cruise_duration_hours = 2 # Example cruise duration
    cruise_distance_change = horizontal_speed_kts * cruise_duration_hours * 6076.12 # Convert knots to feet/hour, then to nautical miles
    
    # Add points for cruise to make it visible
    num_cruise_segments = 5
    for i in range(1, num_cruise_segments + 1):
        path.append((current_distance + (cruise_distance_change / num_cruise_segments) * i, cruise_altitude))
        speeds.append(get_profile_value(climb_profile, cruise_altitude, horizontal_speed_kts)) # Use cruise speed

    current_distance += cruise_distance_change

    # Descent to Landing
    while current_altitude > landing_altitude:
        path.append((current_distance, current_altitude))
        
        target_speed = get_profile_value(descent_profile, current_altitude, horizontal_speed_kts)
        vertical_speed_fpm = get_profile_value(descent_vs_profile, current_altitude, 1000) # feet per minute

        speeds.append(target_speed)

        vertical_speed_fps = vertical_speed_fpm / 60
        
        altitude_change = -vertical_speed_fps * time_step_seconds # Negative for descent
        current_altitude += altitude_change

        horizontal_speed_fps = horizontal_speed_kts * 1.68781
        distance_change = horizontal_speed_fps * time_step_seconds
        current_distance += distance_change

        if current_altitude <= landing_altitude:
            current_altitude = landing_altitude # Cap at landing altitude
            path.append((current_distance, current_altitude))
            speeds.append(target_speed) # Add speed for the landing point
            break
            
    return np.array(path), np.array(speeds)

def plot_flight_profile(path, speeds):
    """Plots the flight profile with speed annotations."""
    distances = path[:, 0] / 6076.12 # Convert feet to nautical miles for x-axis
    altitudes = path[:, 1] # Altitudes are already in feet

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(distances, altitudes, marker='o', linestyle='-', color='blue', markersize=4)

    # Annotate speeds on segments
    for i in range(len(distances) - 1):
        mid_x = (distances[i] + distances[i+1]) / 2
        mid_y = (altitudes[i] + altitudes[i+1]) / 2
        
        # Only annotate if there's a significant change or it's a new segment
        if i == 0 or speeds[i] != speeds[i-1] or speeds[i+1] != speeds[i]:
            ax.text(mid_x, mid_y + 500, f"{int(speeds[i])} kts", 
                    fontsize=8, color='red', ha='center', va='bottom')

    ax.set_title("Typical Flight Vertical Profile (Narrow-body Jet)")
    ax.set_xlabel("Horizontal Distance (Nautical Miles)")
    ax.set_ylabel("Altitude (Feet)")
    ax.grid(True)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    flight_path, segment_speeds = generate_flight_path(
        NARROW_BODY_JET_CLIMB_PROFILE,
        NARROW_BODY_JET_CLIMB_VS_PROFILE,
        NARROW_BODY_JET_DESCENT_PROFILE,
        NARROW_BODY_JET_DESCENT_VS_PROFILE,
    )
    plot_flight_profile(flight_path, segment_speeds)
