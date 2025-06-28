#!/usr/bin/env python3
"""
Flight Vertical Profile Visualization

This script creates a vertical profile plot showing climb, cruise, and descent phases
of a typical flight based on the profiles defined in vnav_profiles_rev1.py.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple, Optional

# Set seaborn style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (8, 6)

def interpolate_profile_value(altitude_ft: float, profile: List[Tuple[float, float]]) -> float:
    """
    Interpolate a value from a profile based on altitude.
    
    Args:
        altitude_ft: Current altitude in feet
        profile: List of (altitude_threshold, value) tuples
    
    Returns:
        Interpolated value
    """
    for i, (alt_threshold, value) in enumerate(profile):
        if altitude_ft <= alt_threshold:
            if i == 0:
                return value
            # Linear interpolation between segments
            prev_alt, prev_value = profile[i-1]
            if prev_alt == float('inf'):
                return prev_value
            
            # Handle infinite altitude threshold
            if alt_threshold == float('inf'):
                return value
            
            # Linear interpolation
            ratio = (altitude_ft - prev_alt) / (alt_threshold - prev_alt)
            return prev_value + ratio * (value - prev_value)
    
    # If altitude is above all thresholds, return the last value
    return profile[-1][1]

def create_flight_profile(aircraft_type: str = "NARROW_BODY_JET",
                         cruise_altitude: float = 35000,
                         total_distance_nm: float = 800,
                         plot_title: Optional[str] = None) -> None:
    """
    Create a vertical flight profile plot.
    
    Args:
        aircraft_type: Type of aircraft (NARROW_BODY_JET, WIDE_BODY_JET, etc.)
        cruise_altitude: Cruise altitude in feet
        total_distance_nm: Total flight distance in nautical miles
        plot_title: Custom title for the plot
    """
    
    # Import profiles from the vnav module
    from src.equinox.vnav.vnav_profiles_rev1 import (
        NARROW_BODY_JET_CLIMB_PROFILE, NARROW_BODY_JET_DESCENT_PROFILE,
        NARROW_BODY_JET_CLIMB_VS_PROFILE, NARROW_BODY_JET_DESCENT_VS_PROFILE,
        WIDE_BODY_JET_CLIMB_PROFILE, WIDE_BODY_JET_DESCENT_PROFILE,
        WIDE_BODY_JET_CLIMB_VS_PROFILE, WIDE_BODY_JET_DESCENT_VS_PROFILE,
        BUSINESS_JET_CLIMB_PROFILE, BUSINESS_JET_DESCENT_PROFILE,
        BUSINESS_JET_CLIMB_VS_PROFILE, BUSINESS_JET_DESCENT_VS_PROFILE
    )
    
    # Select profiles based on aircraft type
    profiles = {
        "NARROW_BODY_JET": {
            "climb_speed": NARROW_BODY_JET_CLIMB_PROFILE,
            "descent_speed": NARROW_BODY_JET_DESCENT_PROFILE,
            "climb_vs": NARROW_BODY_JET_CLIMB_VS_PROFILE,
            "descent_vs": NARROW_BODY_JET_DESCENT_VS_PROFILE,
            "cruise_speed": 460
        },
        "WIDE_BODY_JET": {
            "climb_speed": WIDE_BODY_JET_CLIMB_PROFILE,
            "descent_speed": WIDE_BODY_JET_DESCENT_PROFILE,
            "climb_vs": WIDE_BODY_JET_CLIMB_VS_PROFILE,
            "descent_vs": WIDE_BODY_JET_DESCENT_VS_PROFILE,
            "cruise_speed": 490
        },
        "BUSINESS_JET": {
            "climb_speed": BUSINESS_JET_CLIMB_PROFILE,
            "descent_speed": BUSINESS_JET_DESCENT_PROFILE,
            "climb_vs": BUSINESS_JET_CLIMB_VS_PROFILE,
            "descent_vs": BUSINESS_JET_DESCENT_VS_PROFILE,
            "cruise_speed": 480
        }
    }
    
    profile = profiles[aircraft_type]
    
    # Flight phases parameters
    climb_distance_nm = 150  # Distance to climb
    descent_distance_nm = 150  # Distance to descend
    cruise_distance_nm = total_distance_nm - climb_distance_nm - descent_distance_nm
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Initialize arrays for the profile
    distances = []
    altitudes = []
    speeds = []
    phases = []
    
    # CLIMB PHASE
    print("Computing climb phase...")
    current_distance = 0
    current_altitude = 0
    
    climb_segments = []
    altitude_step = 2000  # Altitude increments for visualization
    
    while current_altitude < cruise_altitude:
        segment_start_alt = current_altitude
        segment_end_alt = min(current_altitude + altitude_step, cruise_altitude)
        
        # Get vertical speed for this altitude range
        vs_fpm = interpolate_profile_value(segment_end_alt, profile["climb_vs"])
        speed_kts = interpolate_profile_value(segment_end_alt, profile["climb_speed"])
        
        # Calculate time to climb this segment (in minutes)
        altitude_gain = segment_end_alt - segment_start_alt
        time_minutes = altitude_gain / vs_fpm
        
        # Calculate distance covered during climb
        segment_distance = (speed_kts * time_minutes) / 60  # Convert to nautical miles
        
        # Add segment points
        distances.extend([current_distance, current_distance + segment_distance])
        altitudes.extend([segment_start_alt, segment_end_alt])
        speeds.extend([speed_kts, speed_kts])
        phases.extend(['climb', 'climb'])
        
        climb_segments.append({
            'start_dist': current_distance,
            'end_dist': current_distance + segment_distance,
            'start_alt': segment_start_alt,
            'end_alt': segment_end_alt,
            'speed': speed_kts,
            'vs': vs_fpm
        })
        
        current_distance += segment_distance
        current_altitude = segment_end_alt
    
    # CRUISE PHASE
    print("Computing cruise phase...")
    cruise_start_distance = current_distance
    cruise_end_distance = cruise_start_distance + cruise_distance_nm
    
    distances.extend([cruise_start_distance, cruise_end_distance])
    altitudes.extend([cruise_altitude, cruise_altitude])
    speeds.extend([profile["cruise_speed"], profile["cruise_speed"]])
    phases.extend(['cruise', 'cruise'])
    
    cruise_segment = {
        'start_dist': cruise_start_distance,
        'end_dist': cruise_end_distance,
        'start_alt': cruise_altitude,
        'end_alt': cruise_altitude,
        'speed': profile["cruise_speed"],
        'vs': 0
    }
    
    # DESCENT PHASE
    print("Computing descent phase...")
    current_distance = cruise_end_distance
    current_altitude = cruise_altitude
    
    descent_segments = []
    
    while current_altitude > 0:
        segment_start_alt = current_altitude
        segment_end_alt = max(current_altitude - altitude_step, 0)
        
        # Get vertical speed for this altitude range
        vs_fpm = interpolate_profile_value(segment_start_alt, profile["descent_vs"])
        speed_kts = interpolate_profile_value(segment_start_alt, profile["descent_speed"])
        
        # Calculate time to descend this segment (in minutes)
        altitude_loss = segment_start_alt - segment_end_alt
        time_minutes = altitude_loss / vs_fpm
        
        # Calculate distance covered during descent
        segment_distance = (speed_kts * time_minutes) / 60  # Convert to nautical miles
        
        # Add segment points
        distances.extend([current_distance, current_distance + segment_distance])
        altitudes.extend([segment_start_alt, segment_end_alt])
        speeds.extend([speed_kts, speed_kts])
        phases.extend(['descent', 'descent'])
        
        descent_segments.append({
            'start_dist': current_distance,
            'end_dist': current_distance + segment_distance,
            'start_alt': segment_start_alt,
            'end_alt': segment_end_alt,
            'speed': speed_kts,
            'vs': vs_fpm
        })
        
        current_distance += segment_distance
        current_altitude = segment_end_alt
    
    # Convert to numpy arrays
    distances = np.array(distances)
    altitudes = np.array(altitudes)
    
    # Plot the profile
    ax.plot(distances, altitudes, linewidth=3, color='steelblue', alpha=0.8)
    
    # Add phase colors
    phase_colors = {'climb': 'lightgreen', 'cruise': 'lightblue', 'descent': 'lightcoral'}
    
    # Fill areas under curve for each phase
    climb_mask = np.array(phases) == 'climb'
    cruise_mask = np.array(phases) == 'cruise'
    descent_mask = np.array(phases) == 'descent'
    
    if np.any(climb_mask):
        ax.fill_between(distances[climb_mask], 0, altitudes[climb_mask], 
                       alpha=0.3, color=phase_colors['climb'], label='Climb')
    
    if np.any(cruise_mask):
        ax.fill_between(distances[cruise_mask], 0, altitudes[cruise_mask], 
                       alpha=0.3, color=phase_colors['cruise'], label='Cruise')
    
    if np.any(descent_mask):
        ax.fill_between(distances[descent_mask], 0, altitudes[descent_mask], 
                       alpha=0.3, color=phase_colors['descent'], label='Descent')
    
    # Add speed annotations for climb segments
    for segment in climb_segments[::2]:  # Every other segment to avoid clutter
        mid_dist = (segment['start_dist'] + segment['end_dist']) / 2
        mid_alt = (segment['start_alt'] + segment['end_alt']) / 2
        ax.annotate(f"{segment['speed']:.0f} kts\n{segment['vs']:.0f} fpm", 
                   xy=(mid_dist, mid_alt), xytext=(10, 10), 
                   textcoords='offset points', fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                   ha='center')
    
    # Add cruise speed annotation
    cruise_mid_dist = (cruise_segment['start_dist'] + cruise_segment['end_dist']) / 2
    ax.annotate(f"{cruise_segment['speed']:.0f} kts\nCruise", 
               xy=(cruise_mid_dist, cruise_altitude), xytext=(0, 20), 
               textcoords='offset points', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
               ha='center')
    
    # Add speed annotations for descent segments
    for segment in descent_segments[::2]:  # Every other segment to avoid clutter
        mid_dist = (segment['start_dist'] + segment['end_dist']) / 2
        mid_alt = (segment['start_alt'] + segment['end_alt']) / 2
        ax.annotate(f"{segment['speed']:.0f} kts\n{segment['vs']:.0f} fpm", 
                   xy=(mid_dist, mid_alt), xytext=(10, 10), 
                   textcoords='offset points', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                   ha='center')
    
    # Customize the plot
    ax.set_xlabel('Distance (Nautical Miles)', fontsize=12)
    ax.set_ylabel('Altitude (Feet)', fontsize=12)
    
    if plot_title:
        ax.set_title(plot_title, fontsize=14, fontweight='bold')
    else:
        # ax.set_title(f'{aircraft_type.replace("_", " ").title()} Flight Profile\n'
        #             f'Cruise Altitude: {cruise_altitude:,.0f} ft, Total Distance: {total_distance_nm:.0f} nm', 
        #             fontsize=14, fontweight='bold')
        pass
    
    # Format y-axis to show altitude in thousands
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Set axis limits
    ax.set_xlim(0, max(distances) * 1.05)
    ax.set_ylim(0, max(altitudes) * 1.1)
    
    plt.tight_layout()
    
    # Print summary
    print(f"\nFlight Profile Summary:")
    print(f"Aircraft Type: {aircraft_type}")
    print(f"Total Distance: {total_distance_nm:.0f} nm")
    print(f"Cruise Altitude: {cruise_altitude:,.0f} ft")
    print(f"Climb Distance: {climb_segments[-1]['end_dist']:.0f} nm")
    print(f"Cruise Distance: {cruise_distance_nm:.0f} nm") 
    print(f"Descent Distance: {descent_segments[-1]['end_dist'] - cruise_end_distance:.0f} nm")
    
    return fig, ax

def main():
    """Main function to create multiple flight profile plots."""
    
    # Create plots for different aircraft types
    aircraft_types = ["NARROW_BODY_JET", "WIDE_BODY_JET", "BUSINESS_JET"]
    
    for aircraft_type in aircraft_types:
        print(f"\nCreating plot for {aircraft_type}...")
        
        # Adjust parameters based on aircraft type
        if aircraft_type == "NARROW_BODY_JET":
            cruise_alt = 35000
            distance = 800
        elif aircraft_type == "WIDE_BODY_JET":
            cruise_alt = 41000
            distance = 1200
        else:  # BUSINESS_JET
            cruise_alt = 45000
            distance = 1000
        
        fig, ax = create_flight_profile(
            aircraft_type=aircraft_type,
            cruise_altitude=cruise_alt,
            total_distance_nm=distance,
            plot_title=None
        )
        
        # Save the plot
        filename = f"flight_profile_{aircraft_type.lower()}.pdf"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")
        
        # Show the plot
        plt.show()

if __name__ == "__main__":
    main()