from equinox.wind.wind_model import WindModel
from datetime import datetime
import os


def test_wind_model():
    # Get the era5 data directory
    # Get the directory of the current file
    current_dir = os.path.dirname(__file__)
    # Navigate up one directory to the project root (assuming tests directory is directly under the root)
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    # Construct the data_dir path from the project root
    data_dir = os.path.join(project_root, "data", "era5")

    wind_model = WindModel(date_str="2024-04-01", data_dir=data_dir)
    u, v = wind_model.get_wind_components(
        lat=48.8566, lon=2.3522, alt_ft=50.0, time=datetime(2024, 4, 1, 12, 0, 0)
    )
    assert u == u  # Check if u is not NaN
    assert v == v  # Check if v is not NaN
    print("Wind at 48.8566N, 2.3522E (Paris), 50ft, 12:00:00 is", u, v)

    u, v = wind_model.get_wind_components(
        lat=48.8566, lon=2.3522, alt_ft=1000.0, time=datetime(2024, 4, 1, 12, 0, 0)
    )
    assert u == u  # Check if u is not NaN
    assert v == v  # Check if v is not NaN
    print("Wind at 48.8566N, 2.3522E (Paris), 1000ft, 12:00:00 is", u, v)

    u, v = wind_model.get_wind_components(
        lat=36.7213, lon=-4.4214, alt_ft=30000.0, time=datetime(2024, 4, 1, 12, 0, 0)
    )
    assert u == u  # Check if u is not NaN
    assert v == v  # Check if v is not NaN
    print("Wind at 36.7213N, -4.4214E (Malaga), 30000ft, 12:00:00 is", u, v)

    u, v = wind_model.get_wind_components(
        lat=48.8566, lon=2.3522, alt_ft=30000.0, time=datetime(2024, 4, 1, 12, 0, 0)
    )
    assert u == u  # Check if u is not NaN
    assert v == v  # Check if v is not NaN
    print("Wind at 48.8566N, 2.3522E (Paris), 30000ft, 12:00:00 is", u, v)


def test_cape_cin():
    # Get the era5 data directory
    # Get the directory of the current file
    current_dir = os.path.dirname(__file__)
    # Navigate up one directory to the project root (assuming tests directory is directly under the root)
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    # Construct the data_dir path from the project root
    data_dir = os.path.join(project_root, "data", "era5")
    wind_model = WindModel(date_str="2024-04-01", data_dir=data_dir)
    cape, cin = wind_model.get_cape_cin(
        lat=48.8566, lon=2.3522, time=datetime(2024, 4, 1, 12, 0, 0)
    )
    assert cape == cape  # Check if cape is not NaN
    # assert cin == cin # Check if cin is not NaN, we skip this because cin is not available in the dummy data
    print("CAPE/CIN at 48.8566N, 2.3522E (Paris), 12:00:00 is", cape, cin)


if __name__ == "__main__":
    test_wind_model()
    test_cape_cin()
