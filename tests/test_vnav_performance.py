from equinox.vnav.vnav_performance import Performance
from equinox.vnav.vnav_profiles_rev1 import (
    NARROW_BODY_JET_CLIMB_PROFILE,
    NARROW_BODY_JET_CLIMB_VS_PROFILE,
    NARROW_BODY_JET_DESCENT_PROFILE,
    NARROW_BODY_JET_DESCENT_VS_PROFILE,
)


def sec2Formatted(seconds: float) -> str:
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}:{seconds:02d}"


# Only functions starting with 'test_' will be tested by pytest
def test_narrow_body_jet():
    # Case 1: an ordinary narrow-body jet
    print("Case 1: an ordinary narrow-body jet")
    flight_performance = Performance(
        climb_speed_profile=NARROW_BODY_JET_CLIMB_PROFILE,
        descent_speed_profile=NARROW_BODY_JET_DESCENT_PROFILE,
        climb_vertical_speed_profile=NARROW_BODY_JET_CLIMB_VS_PROFILE,
        descent_vertical_speed_profile=NARROW_BODY_JET_DESCENT_VS_PROFILE,
        cruise_altitude_ft=35000,
        cruise_speed_kts=450,
    )

    # Get climb ETAs
    origin_airport_elevation_ft = 0
    climb_etas = flight_performance.get_climb_eta(
        origin_airport_elevation_ft=origin_airport_elevation_ft
    )
    # example climb_etas: [(0.0, 0.0), (10000, 218.1818181818182), (28000, 818.1818181818182), (35000, 1168.1818181818182)] (alt, time in seconds)
    print(f"{'Alt':<10} {'Time (min)':<10}")
    print("-" * 20)
    for alt, time in climb_etas:
        print(f"{alt:<10} {sec2Formatted(time):<10}")

    # Assert last climb time is within 17-19 minutes (1020-1140 seconds)
    assert 1020 <= climb_etas[-1][1] <= 1140

    # Get descent ETAs
    destination_airport_elevation_ft = 0
    descent_etas = flight_performance.get_descent_eta(
        destination_airport_elevation_ft=destination_airport_elevation_ft
    )
    print(f"{'Alt':<10} {'Time (min)':<10}")
    print("-" * 20)
    for alt, time in descent_etas:
        print(f"{alt:<10} {sec2Formatted(time):<10}")

    # Assert last descent time is within 21-23 minutes (1260-1380 seconds)
    assert 1260 <= descent_etas[-1][1] <= 1380

def test_narrow_body_jet_distance():
    flight_performance = Performance(
        climb_speed_profile=NARROW_BODY_JET_CLIMB_PROFILE,
        descent_speed_profile=NARROW_BODY_JET_DESCENT_PROFILE,
        climb_vertical_speed_profile=NARROW_BODY_JET_CLIMB_VS_PROFILE,
        descent_vertical_speed_profile=NARROW_BODY_JET_DESCENT_VS_PROFILE,
        cruise_altitude_ft=35000,
        cruise_speed_kts=450,
    )

    origin_airport_elevation_ft = 0
    along_track_wind_adjusted_distance = flight_performance.get_along_track_wind_adjusted_distance(
        origin_airport_elevation_ft=origin_airport_elevation_ft
    ) # example: [(0.0, 0.0), (10000, 13.888888888888888), (20000, 41.388888888888886), (28000, 67.38888888888889), (35000, 112.11111111111111)] (alt (ft), distance (nm))
    print(f"{'Alt (ft)':<10} {'Distance (nm)':<15}")
    print("-" * 25)
    for alt, dist in along_track_wind_adjusted_distance:
        print(f"{alt:<10} {dist:<15.2f}")

if __name__ == '__main__':
    # test_narrow_body_jet()
    test_narrow_body_jet_distance()