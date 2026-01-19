import pandas as pd
from typing import List, Dict
from scipy.interpolate import interp1d

from equinox.helpers.haversine import haversine, bearing, destination_point
from equinox.vnav.vnav_performance import (
    Performance,
    get_eta_and_distance_climb,
    get_eta_and_distance_descent,
)


def time_str_to_seconds(time_str: str) -> int:
    """Convert HHMMSS string or int to seconds since midnight."""
    time_str = str(time_str).zfill(6)
    h = int(time_str[0:2])
    m = int(time_str[2:4])
    s = int(time_str[4:6])
    return h * 3600 + m * 60 + s


def seconds_to_time_str(seconds: int) -> str:
    """Convert seconds since midnight to HHMMSS string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}{m:02d}{s:02d}"


class VerticalTranchifier:
    def __init__(self, performance: Performance, tranche_altitudes: List[float]):
        self.performance = performance
        self.tranche_altitudes = sorted(tranche_altitudes)
        self._create_interpolators()

    def _create_interpolators(self) -> None:
        origin_elevation_ft = 0
        destination_elevation_ft = 0

        climb_profile = get_eta_and_distance_climb(self.performance, origin_elevation_ft)
        descent_profile = get_eta_and_distance_descent(self.performance, destination_elevation_ft)

        climb_alts = [p[0] for p in climb_profile]
        climb_times = [p[1] for p in climb_profile]
        climb_dists = [p[2] for p in climb_profile]
        self.climb_time_interp = interp1d(climb_alts, climb_times, bounds_error=False, fill_value="extrapolate")
        self.climb_dist_interp = interp1d(climb_alts, climb_dists, bounds_error=False, fill_value="extrapolate")

        if not descent_profile:
            self.descent_time_interp = interp1d([], [], bounds_error=False, fill_value=0)
            self.descent_dist_interp = interp1d([], [], bounds_error=False, fill_value=0)
            return

        tod_alt, tod_time_to_go, tod_dist_to_go = descent_profile[0]
        processed = []
        for alt, time_to_go, dist_to_go in descent_profile:
            time_from_tod = time_to_go - tod_time_to_go
            dist_from_tod = tod_dist_to_go - dist_to_go
            processed.append((alt, time_from_tod, dist_from_tod))

        processed.sort(key=lambda p: p[0])
        descent_alts = [p[0] for p in processed]
        descent_times = [p[1] for p in processed]
        descent_dists = [p[2] for p in processed]
        self.descent_time_interp = interp1d(descent_alts, descent_times, bounds_error=False, fill_value="extrapolate")
        self.descent_dist_interp = interp1d(descent_alts, descent_dists, bounds_error=False, fill_value="extrapolate")

    def process_trajectory(self, input_csv_path: str, output_csv_path: str) -> None:
        df = pd.read_csv(input_csv_path)
        if "route" not in df.columns:
            df["route"] = "ROUTE_1"

        new_segments = []
        in_descent_sequence = False

        for _, segment in df.iterrows():
            flight_level_begin = segment["flight_level_begin"] * 100
            flight_level_end = segment["flight_level_end"] * 100

            if flight_level_begin < flight_level_end:
                new_segments.extend(self._tranchify_climb_segment(segment))
                in_descent_sequence = False
            elif flight_level_begin > flight_level_end:
                new_segments.extend(
                    self._tranchify_descent_segment(
                        segment,
                        allow_cruise_before_descent=not in_descent_sequence,
                    )
                )
                in_descent_sequence = True
            else:
                new_segments.append(segment.to_dict())
                in_descent_sequence = False

        new_df = pd.DataFrame(new_segments)
        new_df.to_csv(output_csv_path, index=False)
        print(f"Tranchified trajectory saved to {output_csv_path}")

    def _tranchify_climb_segment(self, segment: pd.Series) -> List[Dict]:
        alt_begin_ft = segment["flight_level_begin"] * 100
        alt_end_ft = segment["flight_level_end"] * 100
        segment_route = segment.get("route")

        relevant_tranches = [alt for alt in self.tranche_altitudes if alt_begin_ft < alt < alt_end_ft]
        if not relevant_tranches:
            return [segment.to_dict()]

        lat_begin, lon_begin = segment["latitude_begin"], segment["longitude_begin"]
        lat_end, lon_end = segment["latitude_end"], segment["longitude_end"]
        time_begin_sec = time_str_to_seconds(segment["time_begin_segment"])
        time_end_sec = time_str_to_seconds(segment["time_end_segment"])

        segment_bearing = bearing((lat_begin, lon_begin), (lat_end, lon_end))
        total_ground_dist_nm = haversine(lat_begin, lon_begin, lat_end, lon_end)
        total_time_sec = time_end_sec - time_begin_sec

        vnav_dist_needed = self.climb_dist_interp(alt_end_ft) - self.climb_dist_interp(alt_begin_ft)
        vnav_time_needed = self.climb_time_interp(alt_end_ft) - self.climb_time_interp(alt_begin_ft)

        all_points_alt = [alt_begin_ft] + relevant_tranches + [alt_end_ft]
        new_segments = []
        last_point = {"alt": alt_begin_ft, "lat": lat_begin, "lon": lon_begin, "time": time_begin_sec}

        can_insert_post_cruise = False
        try:
            dist_needed = float(vnav_dist_needed)
            time_needed = float(vnav_time_needed)
        except Exception:
            dist_needed = vnav_dist_needed
            time_needed = vnav_time_needed

        if (total_ground_dist_nm > dist_needed) and (total_time_sec > time_needed):
            cruise_dist_post = total_ground_dist_nm - dist_needed
            cruise_time_post = total_time_sec - time_needed
            if cruise_dist_post > 0 and cruise_time_post > 0:
                can_insert_post_cruise = True

        if can_insert_post_cruise:
            for i in range(1, len(all_points_alt)):
                current_alt = all_points_alt[i]
                vnav_dist_from_start = self.climb_dist_interp(current_alt) - self.climb_dist_interp(alt_begin_ft)
                vnav_time_from_start = self.climb_time_interp(current_alt) - self.climb_time_interp(alt_begin_ft)

                current_lat, current_lon = destination_point(
                    (lat_begin, lon_begin), segment_bearing, vnav_dist_from_start
                )
                current_time = time_begin_sec + vnav_time_from_start

                new_segments.append({
                    "segment_identifier": f"SYNTHETIC_CLIMB_{i-1}",
                    "origin_aerodrome": segment["origin_aerodrome"],
                    "destination_aerodrome": segment["destination_aerodrome"],
                    "time_begin_segment": seconds_to_time_str(round(last_point["time"])),
                    "time_end_segment": seconds_to_time_str(round(current_time)),
                    "flight_level_begin": last_point["alt"] / 100,
                    "flight_level_end": current_alt / 100,
                    "latitude_begin": last_point["lat"],
                    "longitude_begin": last_point["lon"],
                    "latitude_end": current_lat,
                    "longitude_end": current_lon,
                    "flight_identifier": segment["flight_identifier"],
                    "route": segment_route,
                })
                last_point = {"alt": current_alt, "lat": current_lat, "lon": current_lon, "time": current_time}

            toc_point = last_point
            new_segments.append({
                "segment_identifier": "SYNTHETIC_CRUISE_AFTER_CLIMB",
                "origin_aerodrome": segment["origin_aerodrome"],
                "destination_aerodrome": segment["destination_aerodrome"],
                "time_begin_segment": seconds_to_time_str(round(toc_point["time"])),
                "time_end_segment": seconds_to_time_str(time_end_sec),
                "flight_level_begin": toc_point["alt"] / 100,
                "flight_level_end": toc_point["alt"] / 100,
                "latitude_begin": toc_point["lat"],
                "longitude_begin": toc_point["lon"],
                "latitude_end": lat_end,
                "longitude_end": lon_end,
                "flight_identifier": segment["flight_identifier"],
                "route": segment_route,
            })
        else:
            for i in range(1, len(all_points_alt)):
                current_alt = all_points_alt[i]
                vnav_dist_from_start = self.climb_dist_interp(current_alt) - self.climb_dist_interp(alt_begin_ft)
                vnav_time_from_start = self.climb_time_interp(current_alt) - self.climb_time_interp(alt_begin_ft)

                dist_fraction = vnav_dist_from_start / vnav_dist_needed if vnav_dist_needed > 0 else 0
                time_fraction = vnav_time_from_start / vnav_time_needed if vnav_time_needed > 0 else 0

                current_dist_nm = total_ground_dist_nm * dist_fraction
                current_lat, current_lon = destination_point(
                    (lat_begin, lon_begin), segment_bearing, current_dist_nm
                )
                current_time = time_begin_sec + total_time_sec * time_fraction

                new_segments.append({
                    "segment_identifier": f"SYNTHETIC_CLIMB_{i-1}",
                    "origin_aerodrome": segment["origin_aerodrome"],
                    "destination_aerodrome": segment["destination_aerodrome"],
                    "time_begin_segment": seconds_to_time_str(round(last_point["time"])),
                    "time_end_segment": seconds_to_time_str(round(current_time)),
                    "flight_level_begin": last_point["alt"] / 100,
                    "flight_level_end": current_alt / 100,
                    "latitude_begin": last_point["lat"],
                    "longitude_begin": last_point["lon"],
                    "latitude_end": current_lat,
                    "longitude_end": current_lon,
                    "flight_identifier": segment["flight_identifier"],
                    "route": segment_route,
                })
                last_point = {"alt": current_alt, "lat": current_lat, "lon": current_lon, "time": current_time}

        return new_segments

    def _tranchify_descent_segment(self, segment: pd.Series, *, allow_cruise_before_descent: bool = True) -> List[Dict]:
        alt_begin_ft = segment["flight_level_begin"] * 100
        alt_end_ft = segment["flight_level_end"] * 100
        segment_route = segment.get("route")

        relevant_tranches = [alt for alt in self.tranche_altitudes if alt_end_ft < alt < alt_begin_ft]
        if not relevant_tranches:
            return [segment.to_dict()]

        lat_begin, lon_begin = segment["latitude_begin"], segment["longitude_begin"]
        lat_end, lon_end = segment["latitude_end"], segment["longitude_end"]
        time_begin_sec = time_str_to_seconds(segment["time_begin_segment"])
        time_end_sec = time_str_to_seconds(segment["time_end_segment"])

        segment_bearing = bearing((lat_begin, lon_begin), (lat_end, lon_end))
        total_ground_dist_nm = haversine(lat_begin, lon_begin, lat_end, lon_end)
        total_time_sec = time_end_sec - time_begin_sec

        vnav_dist_needed = self.descent_dist_interp(alt_end_ft) - self.descent_dist_interp(alt_begin_ft)
        vnav_time_needed = self.descent_time_interp(alt_end_ft) - self.descent_time_interp(alt_begin_ft)

        all_points_alt = [alt_begin_ft] + sorted(relevant_tranches, reverse=True) + [alt_end_ft]
        new_segments = []

        can_insert_pre_cruise = False
        if allow_cruise_before_descent:
            try:
                dist_needed = float(vnav_dist_needed)
                time_needed = float(vnav_time_needed)
            except Exception:
                dist_needed = vnav_dist_needed
                time_needed = vnav_time_needed
            if (total_ground_dist_nm > dist_needed) and (total_time_sec > time_needed):
                cruise_dist = total_ground_dist_nm - dist_needed
                cruise_time = total_time_sec - time_needed
                if cruise_dist > 0 and cruise_time > 0:
                    can_insert_pre_cruise = True

        if can_insert_pre_cruise:
            cruise_dist = total_ground_dist_nm - vnav_dist_needed
            cruise_time = total_time_sec - vnav_time_needed

            tod_lat, tod_lon = destination_point((lat_begin, lon_begin), segment_bearing, cruise_dist)
            tod_time = time_begin_sec + cruise_time

            new_segments.append({
                "segment_identifier": "SYNTHETIC_CRUISE_BEFORE_DESCENT",
                "origin_aerodrome": segment["origin_aerodrome"],
                "destination_aerodrome": segment["destination_aerodrome"],
                "time_begin_segment": seconds_to_time_str(time_begin_sec),
                "time_end_segment": seconds_to_time_str(round(tod_time)),
                "flight_level_begin": alt_begin_ft / 100,
                "flight_level_end": alt_begin_ft / 100,
                "latitude_begin": lat_begin,
                "longitude_begin": lon_begin,
                "latitude_end": tod_lat,
                "longitude_end": tod_lon,
                "flight_identifier": segment["flight_identifier"],
                "route": segment_route,
            })

            last_point = {"alt": alt_begin_ft, "lat": tod_lat, "lon": tod_lon, "time": tod_time}
            for i in range(1, len(all_points_alt)):
                current_alt = all_points_alt[i]
                vnav_dist_covered = self.descent_dist_interp(current_alt) - self.descent_dist_interp(alt_begin_ft)
                vnav_time_elapsed = self.descent_time_interp(current_alt) - self.descent_time_interp(alt_begin_ft)

                current_lat, current_lon = destination_point(
                    (tod_lat, tod_lon), segment_bearing, vnav_dist_covered
                )
                current_time = tod_time + vnav_time_elapsed

                new_segments.append({
                    "segment_identifier": f"SYNTHETIC_DESCENT_{i-1}",
                    "origin_aerodrome": segment["origin_aerodrome"],
                    "destination_aerodrome": segment["destination_aerodrome"],
                    "time_begin_segment": seconds_to_time_str(round(last_point["time"])),
                    "time_end_segment": seconds_to_time_str(round(current_time)),
                    "flight_level_begin": last_point["alt"] / 100,
                    "flight_level_end": current_alt / 100,
                    "latitude_begin": last_point["lat"],
                    "longitude_begin": last_point["lon"],
                    "latitude_end": current_lat,
                    "longitude_end": current_lon,
                    "flight_identifier": segment["flight_identifier"],
                    "route": segment_route,
                })
                last_point = {"alt": current_alt, "lat": current_lat, "lon": current_lon, "time": current_time}
        else:
            last_point = {"alt": alt_begin_ft, "lat": lat_begin, "lon": lon_begin, "time": time_begin_sec}
            for i in range(1, len(all_points_alt)):
                current_alt = all_points_alt[i]
                vnav_dist_covered = self.descent_dist_interp(current_alt) - self.descent_dist_interp(alt_begin_ft)
                vnav_time_elapsed = self.descent_time_interp(current_alt) - self.descent_time_interp(alt_begin_ft)

                dist_fraction = vnav_dist_covered / vnav_dist_needed if vnav_dist_needed > 0 else 0
                time_fraction = vnav_time_elapsed / vnav_time_needed if vnav_time_needed > 0 else 0

                current_dist_nm = total_ground_dist_nm * dist_fraction
                current_lat, current_lon = destination_point(
                    (lat_begin, lon_begin), segment_bearing, current_dist_nm
                )
                current_time = time_begin_sec + total_time_sec * time_fraction

                new_segments.append({
                    "segment_identifier": f"SYNTHETIC_DESCENT_{i-1}",
                    "origin_aerodrome": segment["origin_aerodrome"],
                    "destination_aerodrome": segment["destination_aerodrome"],
                    "time_begin_segment": seconds_to_time_str(round(last_point["time"])),
                    "time_end_segment": seconds_to_time_str(round(current_time)),
                    "flight_level_begin": last_point["alt"] / 100,
                    "flight_level_end": current_alt / 100,
                    "latitude_begin": last_point["lat"],
                    "longitude_begin": last_point["lon"],
                    "latitude_end": current_lat,
                    "longitude_end": current_lon,
                    "flight_identifier": segment["flight_identifier"],
                    "route": segment_route,
                })
                last_point = {"alt": current_alt, "lat": current_lat, "lon": current_lon, "time": current_time}

        return new_segments
