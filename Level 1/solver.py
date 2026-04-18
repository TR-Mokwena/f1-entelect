import json
import math
import sys
from copy import deepcopy

GRAVITY = 9.8
K_STRAIGHT = 0.0000166
K_BRAKING  = 0.0398
K_CORNER   = 0.000265
K_BASE     = 0.0005
K_DRAG     = 0.0000000015

BASE_FRICTION = {
    "Soft":         1.8,
    "Medium":       1.7,
    "Hard":         1.6,
    "Intermediate": 1.2,
    "Wet":          1.1,
}

WEATHER_KEY_MAP = {
    "dry":        ("dry_friction_multiplier",        "dry_degradation"),
    "cold":       ("cold_friction_multiplier",       "cold_degradation"),
    "light_rain": ("light_rain_friction_multiplier", "light_rain_degradation"),
    "heavy_rain": ("heavy_rain_friction_multiplier", "heavy_rain_degradation"),
}


def accel_distance(vi, vf, a):
    if a == 0:
        return 0.0
    return (vf**2 - vi**2) / (2 * a)

def accel_time(vi, vf, a):
    if a == 0:
        return 0.0
    return (vf - vi) / a

def time_at_constant(distance, speed):
    if speed == 0:
        return float('inf')
    return distance / speed

def tyre_friction(compound, weather_condition, tyre_props, total_degradation=0.0):
    friction_key, _ = WEATHER_KEY_MAP[weather_condition]
    base = BASE_FRICTION[compound]
    multiplier = tyre_props[friction_key]
    return (base - total_degradation) * multiplier

def max_corner_speed(friction, radius, crawl_speed):
    return math.sqrt(friction * GRAVITY * radius) + crawl_speed


def simulate_straight(seg, entry_speed, target_speed, brake_start_m_before_next,
                      accel, brake, max_speed, crawl_speed, corner_entry_speed_needed):
    L = seg["length_m"]
    effective_target = max(target_speed, entry_speed)
    effective_target = min(effective_target, max_speed)

    d_accel = accel_distance(entry_speed, effective_target, accel)
    if d_accel > L:
        # straight is too short to reach target — just accelerate the whole way
        vf_no_brake = math.sqrt(entry_speed**2 + 2 * accel * L)
        vf_no_brake = min(vf_no_brake, max_speed)
        t = accel_time(entry_speed, vf_no_brake, accel) if vf_no_brake > entry_speed else (L / entry_speed if entry_speed > 0 else float('inf'))
        return vf_no_brake, t, True

    d_brake = accel_distance(corner_entry_speed_needed, effective_target, brake)
    brake_start_pos = L - brake_start_m_before_next

    if d_brake > brake_start_m_before_next:
        pass

    t_accel = accel_time(entry_speed, effective_target, accel) if effective_target > entry_speed else 0.0

    d_cruise = brake_start_pos - d_accel
    if d_cruise < 0:
        # brake point lands inside the acceleration zone — skip cruise, go straight to braking
        speed_at_brake = math.sqrt(entry_speed**2 + 2 * accel * brake_start_pos)
        speed_at_brake = min(speed_at_brake, max_speed)
        t_accel2 = accel_time(entry_speed, speed_at_brake, accel)
        d_brake_actual = L - brake_start_pos
        exit_speed = math.sqrt(max(0, speed_at_brake**2 - 2 * brake * d_brake_actual))
        t_brake = accel_time(exit_speed, speed_at_brake, brake)
        return exit_speed, t_accel2 + t_brake, True

    t_cruise = time_at_constant(d_cruise, effective_target) if d_cruise >= 0 else 0.0

    d_brake_actual = L - brake_start_pos
    exit_speed = math.sqrt(max(0, effective_target**2 - 2 * brake * d_brake_actual))
    t_brake = accel_time(exit_speed, effective_target, brake) if effective_target > exit_speed else 0.0

    total_time = t_accel + t_cruise + t_brake
    return exit_speed, total_time, True


def simulate_corner(seg, entry_speed, friction, crawl_speed, crash_penalty):
    max_speed_corner = max_corner_speed(friction, seg["radius_m"], crawl_speed)
    crashed = entry_speed > max_speed_corner + 1e-6

    if crashed:
        t = time_at_constant(seg["length_m"], crawl_speed)
        return crawl_speed, t + crash_penalty, True
    else:
        t = time_at_constant(seg["length_m"], entry_speed)
        return entry_speed, t, False


def simulate_race(level, strategy):
    car = level["car"]
    accel = car["accel_m/se2"]
    brake_rate = car["brake_m/se2"]
    max_speed = car["max_speed_m/s"]
    crawl_speed = car["crawl_constant_m/s"]
    limp_speed = car["limp_constant_m/s"]
    crash_penalty = level["race"]["corner_crash_penalty_s"]
    pit_exit_speed = level["race"]["pit_exit_speed_m/s"]

    initial_tyre_id = strategy["initial_tyre_id"]
    compound = get_compound(level, initial_tyre_id)
    weather = get_weather_at_time(level, 0)

    tyre_props = level["tyres"]["properties"][compound]
    friction = tyre_friction(compound, weather, tyre_props, 0.0)

    segments = level["track"]["segments"]
    seg_map = {s["id"]: s for s in segments}

    total_time = 0.0
    current_speed = 0.0
    crashes = 0
    in_crawl = False

    for lap_data in strategy["laps"]:
        lap_segs = lap_data["segments"]

        for i, seg_action in enumerate(lap_segs):
            seg = seg_map[seg_action["id"]]

            if seg["type"] == "straight":
                target = seg_action.get("target_m/s", max_speed)
                brake_start = seg_action.get("brake_start_m_before_next", 0)

                next_corner_entry = _next_corner_max_speed(
                    lap_segs, i, seg_map, friction, crawl_speed)

                exit_speed, t, valid = simulate_straight(
                    seg, current_speed, target, brake_start,
                    accel, brake_rate, max_speed, crawl_speed, next_corner_entry)

                in_crawl = False
                current_speed = exit_speed
                total_time += t

            elif seg["type"] == "corner":
                entry = crawl_speed if in_crawl else current_speed

                exit_speed, t, crashed = simulate_corner(
                    seg, entry, friction, crawl_speed, crash_penalty)

                if crashed:
                    crashes += 1
                    in_crawl = True
                else:
                    in_crawl = False

                current_speed = exit_speed
                total_time += t

        pit = lap_data.get("pit", {})
        if pit.get("enter", False):
            pit_time = level["race"]["base_pit_stop_time_s"]
            if pit.get("tyre_change_set_id"):
                pit_time += level["race"]["pit_tyre_swap_time_s"]
                new_id = pit["tyre_change_set_id"]
                compound = get_compound(level, new_id)
                tyre_props = level["tyres"]["properties"][compound]
                friction = tyre_friction(compound, weather, tyre_props, 0.0)
            refuel = pit.get("fuel_refuel_amount_l", 0)
            if refuel > 0:
                pit_time += refuel / level["race"]["pit_refuel_rate_l/s"]
            total_time += pit_time
            current_speed = pit_exit_speed

    return total_time, crashes


def _next_corner_max_speed(lap_segs, current_idx, seg_map, friction, crawl_speed):
    # chained corners share no braking room between them — must enter at the tightest limit
    speeds = []
    j = current_idx + 1
    while j < len(lap_segs):
        seg = seg_map[lap_segs[j]["id"]]
        if seg["type"] == "corner":
            speeds.append(max_corner_speed(friction, seg["radius_m"], crawl_speed))
            j += 1
        else:
            break
    return min(speeds) if speeds else 0.0


def get_compound(level, tyre_id):
    for tyre_set in level["tyres"]["available_sets"]:
        if tyre_id in tyre_set["ids"]:
            return tyre_set["compound"]
    raise ValueError(f"Tyre ID {tyre_id} not found")


def get_weather_at_time(level, race_time):
    conditions = level["weather"]["conditions"]
    start_id = level["race"].get("starting_weather_condition_id", 1)
    start_idx = next((i for i, c in enumerate(conditions) if c["id"] == start_id), 0)

    t = 0.0
    idx = start_idx
    while True:
        c = conditions[idx % len(conditions)]
        if t + c["duration_s"] > race_time:
            return c["condition"]
        t += c["duration_s"]
        idx += 1


def compute_optimal_brake_start(seg, entry_speed, target_speed, corner_entry_speed,
                                 accel, brake_rate, max_speed, crawl_speed):
    L = seg["length_m"]
    effective_target = min(max(target_speed, entry_speed), max_speed)
    safe_corner_speed = corner_entry_speed - 0.001  # tiny margin to avoid floating-point corner crashes
    d_brake = accel_distance(safe_corner_speed, effective_target, brake_rate)
    d_brake = min(d_brake, L)
    return d_brake


def build_optimal_strategy(level):
    car = level["car"]
    accel = car["accel_m/se2"]
    brake_rate = car["brake_m/se2"]
    max_speed = car["max_speed_m/s"]
    crawl_speed = car["crawl_constant_m/s"]
    num_laps = level["race"]["laps"]

    weather = get_weather_at_time(level, 0)
    friction_key = WEATHER_KEY_MAP[weather][0]
    best_tyre_id, best_compound, best_friction = None, None, -1

    for tyre_set in level["tyres"]["available_sets"]:
        compound = tyre_set["compound"]
        props = level["tyres"]["properties"][compound]
        f = BASE_FRICTION[compound] * props[friction_key]
        if f > best_friction:
            best_friction = f
            best_compound = compound
            best_tyre_id = tyre_set["ids"][0]

    print(f"Selected tyre: {best_compound} (ID {best_tyre_id}), friction={best_friction:.4f}")

    segments = level["track"]["segments"]
    seg_map = {s["id"]: s for s in segments}

    corner_max = {}
    for seg in segments:
        if seg["type"] == "corner":
            corner_max[seg["id"]] = max_corner_speed(best_friction, seg["radius_m"], crawl_speed)

    print("\nCorner max speeds:")
    for sid, spd in corner_max.items():
        print(f"  Seg {sid}: {spd:.2f} m/s")

    def build_lap(lap_num, entry_speed_override=None):
        lap_segments = []
        current_speed = entry_speed_override if entry_speed_override is not None else 0.0

        for i, seg in enumerate(segments):
            if seg["type"] == "straight":
                # consecutive corners share no braking room — must enter at the tightest limit
                chain_speeds = []
                j = i + 1
                while j < len(segments):
                    if segments[j]["type"] == "corner":
                        chain_speeds.append(corner_max[segments[j]["id"]])
                        j += 1
                    else:
                        break
                next_corner_speed = min(chain_speeds) if chain_speeds else 0.0

                brake_start = compute_optimal_brake_start(
                    seg, current_speed, max_speed, next_corner_speed,
                    accel, brake_rate, max_speed, crawl_speed)

                lap_segments.append({
                    "id": seg["id"],
                    "type": "straight",
                    "target_m/s": max_speed,
                    "brake_start_m_before_next": round(brake_start, 4)
                })
                current_speed = next_corner_speed

            elif seg["type"] == "corner":
                lap_segments.append({
                    "id": seg["id"],
                    "type": "corner"
                })

        return lap_segments

    laps = []
    for lap_num in range(1, num_laps + 1):
        entry = 0.0 if lap_num == 1 else None  # lap 1 starts stationary; later laps carry exit speed from the last corner
        lap_segs = build_lap(lap_num, entry_speed_override=entry)
        laps.append({
            "lap": lap_num,
            "segments": lap_segs,
            "pit": {"enter": False}
        })

    return {
        "initial_tyre_id": best_tyre_id,
        "laps": laps
    }


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "level1.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "submission.txt"

    with open(input_file) as f:
        level = json.load(f)

    print(f"=== Entelect Grand Prix Solver - Level 1 ===")
    print(f"Track: {level['track']['name']}")
    print(f"Laps:  {level['race']['laps']}")
    print(f"Car:   max={level['car']['max_speed_m/s']} m/s, accel={level['car']['accel_m/se2']} m/s², brake={level['car']['brake_m/se2']} m/s²\n")

    strategy = build_optimal_strategy(level)

    total_time, crashes = simulate_race(level, strategy)
    time_ref = level["race"]["time_reference_s"]
    base_score = 500_000 * (time_ref / total_time) ** 3

    print(f"\n=== Simulation Results ===")
    print(f"Total race time: {total_time:.2f}s")
    print(f"Reference time:  {time_ref:.2f}s")
    print(f"Crashes:         {crashes}")
    print(f"Base score:      {base_score:,.0f}")

    output = json.dumps(strategy, indent=2)
    with open(output_file, "w") as f:
        f.write(output)
    print(f"\nSubmission written to {output_file}")

    print("\n=== Strategy Summary ===")
    for lap in strategy["laps"]:
        print(f"\nLap {lap['lap']}:")
        for seg in lap["segments"]:
            if seg["type"] == "straight":
                print(f"  Seg {seg['id']} [straight]: target={seg['target_m/s']} m/s, brake@{seg['brake_start_m_before_next']:.1f}m before end")
            else:
                print(f"  Seg {seg['id']} [corner]")


if __name__ == "__main__":
    main()
