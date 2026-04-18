"""
Entelect Grand Prix - Level 1 Solver
Optimizes: target speed per straight, braking point, tyre compound selection
No fuel limits, no tyre degradation in Level 1.
"""

import json
import math
import sys
from copy import deepcopy

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# Physics helpers
# ─────────────────────────────────────────────

def accel_distance(vi, vf, a):
    """Distance to accelerate from vi to vf at rate a."""
    if a == 0:
        return 0.0
    return (vf**2 - vi**2) / (2 * a)

def accel_time(vi, vf, a):
    """Time to change speed from vi to vf at rate a."""
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


# ─────────────────────────────────────────────
# Straight segment simulation
# ─────────────────────────────────────────────

def simulate_straight(seg, entry_speed, target_speed, brake_start_m_before_next,
                      accel, brake, max_speed, crawl_speed, corner_entry_speed_needed):
    """
    Simulate a straight segment.
    Returns: (exit_speed, time, valid)
    
    The straight has length L.
    - Car enters at entry_speed
    - Accelerates to min(target_speed, max_speed)
    - Holds that speed until brake_start_m_before_next from the end
    - Brakes from there to corner_entry_speed_needed
    
    If target_speed < entry_speed, car just continues at entry_speed (assumption 11).
    """
    L = seg["length_m"]
    effective_target = max(target_speed, entry_speed)  # assumption 11
    effective_target = min(effective_target, max_speed)

    # Phase 1: accelerate from entry_speed to effective_target
    d_accel = accel_distance(entry_speed, effective_target, accel)
    if d_accel > L:
        # Can't reach target speed — find actual speed at end of straight before braking
        # We'll compute how fast we get and see if we can still brake in time
        # Actually: first check if we need to brake at all
        # Speed after accelerating entire straight
        vf_no_brake = math.sqrt(entry_speed**2 + 2 * accel * L)
        vf_no_brake = min(vf_no_brake, max_speed)
        # Check if we can brake to corner speed from here — but this straight is done
        # For braking: it happens AT the end of this straight (brake_start_m = 0)
        # We'll just return this speed as exit, let caller handle it
        # Time: entire straight is acceleration
        t = accel_time(entry_speed, vf_no_brake, accel) if vf_no_brake > entry_speed else (L / entry_speed if entry_speed > 0 else float('inf'))
        return vf_no_brake, t, True

    # Phase 2: cruise at effective_target
    d_brake = accel_distance(corner_entry_speed_needed, effective_target, brake)  # distance to brake from target to corner speed
    
    # Brake start is measured from the end of the straight
    # We place the braking point at brake_start_m_before_next from segment end
    brake_start_pos = L - brake_start_m_before_next  # position from start of straight

    # Validate: enough room to brake?
    if d_brake > brake_start_m_before_next:
        # Not enough braking distance — car will enter corner too fast
        # We'll flag this but still compute time (penalty applied by grader)
        pass

    # Compute positions of each phase:
    # [0 .. d_accel]: accelerate
    # [d_accel .. brake_start_pos]: cruise
    # [brake_start_pos .. L]: brake

    t_accel = accel_time(entry_speed, effective_target, accel) if effective_target > entry_speed else 0.0
    
    d_cruise = brake_start_pos - d_accel
    if d_cruise < 0:
        # Braking starts before we even finish accelerating
        # Car accelerates for d_accel then brakes immediately
        # Find speed at brake_start_pos
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


# ─────────────────────────────────────────────
# Corner simulation
# ─────────────────────────────────────────────

def simulate_corner(seg, entry_speed, friction, crawl_speed, crash_penalty):
    """
    Returns (exit_speed, time, crashed)
    Corner speed is constant throughout.
    """
    max_speed_corner = max_corner_speed(friction, seg["radius_m"], crawl_speed)
    crashed = entry_speed > max_speed_corner + 1e-6

    if crashed:
        # Car travels at crawl speed
        t = time_at_constant(seg["length_m"], crawl_speed)
        return crawl_speed, t + crash_penalty, True
    else:
        t = time_at_constant(seg["length_m"], entry_speed)
        return entry_speed, t, False


# ─────────────────────────────────────────────
# Full race simulation
# ─────────────────────────────────────────────

def simulate_race(level, strategy):
    """
    Simulate the full race with a given strategy.
    strategy: {
        "initial_tyre_id": int,
        "laps": [
            {
                "lap": int,
                "segments": [
                    {"id": int, "type": "straight", "target_m/s": float, "brake_start_m_before_next": float}
                    {"id": int, "type": "corner"}
                ],
                "pit": {"enter": bool, ...}
            }
        ]
    }
    Returns total_time, list of segment times, crash_count
    """
    car = level["car"]
    accel = car["accel_m/se2"]
    brake_rate = car["brake_m/se2"]
    max_speed = car["max_speed_m/s"]
    crawl_speed = car["crawl_constant_m/s"]
    limp_speed = car["limp_constant_m/s"]
    crash_penalty = level["race"]["corner_crash_penalty_s"]
    pit_exit_speed = level["race"]["pit_exit_speed_m/s"]

    # Resolve tyre
    initial_tyre_id = strategy["initial_tyre_id"]
    compound = get_compound(level, initial_tyre_id)
    weather = get_weather_at_time(level, 0)

    tyre_props = level["tyres"]["properties"][compound]
    friction = tyre_friction(compound, weather, tyre_props, 0.0)

    segments = level["track"]["segments"]
    seg_map = {s["id"]: s for s in segments}

    total_time = 0.0
    current_speed = 0.0  # race starts at 0
    crashes = 0
    in_crawl = False

    for lap_data in strategy["laps"]:
        lap_segs = lap_data["segments"]

        for i, seg_action in enumerate(lap_segs):
            seg = seg_map[seg_action["id"]]
            
            if seg["type"] == "straight":
                target = seg_action.get("target_m/s", max_speed)
                brake_start = seg_action.get("brake_start_m_before_next", 0)

                # Determine the next corner's required entry speed
                # Look ahead for the next corner
                next_corner_entry = _next_corner_max_speed(
                    lap_segs, i, seg_map, friction, crawl_speed)

                exit_speed, t, valid = simulate_straight(
                    seg, current_speed, target, brake_start,
                    accel, brake_rate, max_speed, crawl_speed, next_corner_entry)

                in_crawl = False
                current_speed = exit_speed
                total_time += t

            elif seg["type"] == "corner":
                if in_crawl:
                    entry = crawl_speed
                else:
                    entry = current_speed

                exit_speed, t, crashed = simulate_corner(
                    seg, entry, friction, crawl_speed, crash_penalty)

                if crashed:
                    crashes += 1
                    in_crawl = True
                else:
                    in_crawl = False

                current_speed = exit_speed
                total_time += t

        # Pit stop
        pit = lap_data.get("pit", {})
        if pit.get("enter", False):
            pit_time = level["race"]["base_pit_stop_time_s"]
            if pit.get("tyre_change_set_id"):
                pit_time += level["race"]["pit_tyre_swap_time_s"]
                # Update tyre
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
    """
    Find the required entry speed for consecutive corners after this straight.
    Since there's no braking between chained corners, must enter at the minimum
    max speed across all consecutive corners until the next straight.
    """
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
    # Find starting condition
    start_id = level["race"].get("starting_weather_condition_id", 1)
    # Find index of starting condition
    start_idx = next((i for i, c in enumerate(conditions) if c["id"] == start_id), 0)
    
    t = 0.0
    idx = start_idx
    while True:
        c = conditions[idx % len(conditions)]
        if t + c["duration_s"] > race_time:
            return c["condition"]
        t += c["duration_s"]
        idx += 1


# ─────────────────────────────────────────────
# Strategy optimizer
# ─────────────────────────────────────────────

def compute_optimal_brake_start(seg, entry_speed, target_speed, corner_entry_speed,
                                 accel, brake_rate, max_speed, crawl_speed):
    """
    Compute the optimal braking point: brake as late as possible while still
    slowing to the required corner entry speed.
    Returns brake_start_m_before_next (from end of segment).
    
    The car accelerates to effective_target first, then cruises, then brakes.
    We need to find what speed the car is actually at when it reaches the
    brake_start_pos, and ensure it can decelerate to corner_entry_speed.
    """
    L = seg["length_m"]
    effective_target = min(max(target_speed, entry_speed), max_speed)

    # Apply a tiny safety margin to avoid floating point crashes
    safe_corner_speed = corner_entry_speed - 0.001

    # Distance to brake from effective_target to safe_corner_speed
    d_brake = accel_distance(safe_corner_speed, effective_target, brake_rate)
    # Clamp to segment length
    d_brake = min(d_brake, L)
    return d_brake


def build_optimal_strategy(level):
    """
    Build the optimal Level 1 strategy:
    1. Pick tyre with highest friction for the weather (dry -> Soft)
    2. For each straight: target = max_speed, brake as late as possible
    3. No pit stops needed (tyres don't degrade in L1)
    """
    car = level["car"]
    accel = car["accel_m/se2"]
    brake_rate = car["brake_m/se2"]
    max_speed = car["max_speed_m/s"]
    crawl_speed = car["crawl_constant_m/s"]
    num_laps = level["race"]["laps"]

    # Pick best tyre for weather
    weather = get_weather_at_time(level, 0)
    friction_key = WEATHER_KEY_MAP[weather][0]
    best_tyre_id, best_compound, best_friction = None, None, -1

    for tyre_set in level["tyres"]["available_sets"]:
        compound = tyre_set["compound"]
        props = level["tyres"]["properties"][compound]
        base = BASE_FRICTION[compound]
        multiplier = props[friction_key]
        f = base * multiplier  # no degradation in L1
        if f > best_friction:
            best_friction = f
            best_compound = compound
            best_tyre_id = tyre_set["ids"][0]

    print(f"Selected tyre: {best_compound} (ID {best_tyre_id}), friction={best_friction:.4f}")

    segments = level["track"]["segments"]
    seg_map = {s["id"]: s for s in segments}

    # Pre-compute max corner speeds for each corner
    corner_max = {}
    for seg in segments:
        if seg["type"] == "corner":
            corner_max[seg["id"]] = max_corner_speed(best_friction, seg["radius_m"], crawl_speed)

    print("\nCorner max speeds:")
    for sid, spd in corner_max.items():
        print(f"  Seg {sid}: {spd:.2f} m/s")

    # Build lap template
    def build_lap(lap_num, entry_speed_override=None):
        lap_segments = []
        current_speed = entry_speed_override if entry_speed_override is not None else 0.0
        
        for i, seg in enumerate(segments):
            if seg["type"] == "straight":
                # Find the required exit speed.
                # After this straight, there may be a chain of consecutive corners with
                # NO straight between them. The car cannot brake between chained corners,
                # so we must enter the entire chain at the MINIMUM max speed across all of them.
                next_corner_speed = 0.0
                j = i + 1
                # Walk forward: collect all consecutive corners until the next straight
                chain_speeds = []
                while j < len(segments):
                    if segments[j]["type"] == "corner":
                        chain_speeds.append(corner_max[segments[j]["id"]])
                        j += 1
                    else:
                        break  # hit a straight, stop
                if chain_speeds:
                    next_corner_speed = min(chain_speeds)  # must enter entire chain at this speed

                brake_start = compute_optimal_brake_start(
                    seg, current_speed, max_speed, next_corner_speed,
                    accel, brake_rate, max_speed, crawl_speed)

                lap_segments.append({
                    "id": seg["id"],
                    "type": "straight",
                    "target_m/s": max_speed,
                    "brake_start_m_before_next": round(brake_start, 4)
                })
                # Update current speed for next segment tracking
                # After braking on straight, exit at next_corner_speed
                current_speed = next_corner_speed

            elif seg["type"] == "corner":
                lap_segments.append({
                    "id": seg["id"],
                    "type": "corner"
                })
                # Speed stays constant through corner
                # current_speed unchanged (car enters and exits at same speed)

        return lap_segments

    laps = []
    for lap_num in range(1, num_laps + 1):
        entry = 0.0 if lap_num == 1 else None  # after lap 1, speed carries over from last corner
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


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

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

    # Pretty-print strategy summary
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
