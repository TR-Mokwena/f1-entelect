"""
Entelect Grand Prix - Level 2 Solver
Extends Level 1 with:
  - Full fuel consumption tracking (Kbase + Kdrag formula)
  - Pit stop planning to avoid running out of fuel (limp mode)
  - Fuel bonus optimization: stay as close to soft cap as possible
  - Smart refuel amounts: only fill what's needed to avoid over-cap penalty
"""

import json
import math
import sys
from copy import deepcopy
from itertools import product as iterproduct

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
GRAVITY     = 9.8
K_STRAIGHT  = 0.0000166
K_BRAKING   = 0.0398
K_CORNER    = 0.000265
K_BASE      = 0.0005
K_DRAG      = 0.0000000015

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
    if a == 0:
        return 0.0
    return abs(vf**2 - vi**2) / (2 * a)

def accel_time(vi, vf, a):
    if a == 0:
        return 0.0
    return abs(vf - vi) / a

def time_at_constant(distance, speed):
    if speed <= 0:
        return float('inf')
    return distance / speed

def tyre_friction(compound, weather_condition, tyre_props, total_degradation=0.0):
    friction_key, _ = WEATHER_KEY_MAP[weather_condition]
    base = BASE_FRICTION[compound]
    multiplier = tyre_props[friction_key]
    return (base - total_degradation) * multiplier

def max_corner_speed(friction, radius, crawl_speed):
    return math.sqrt(max(0, friction * GRAVITY * radius)) + crawl_speed


# ─────────────────────────────────────────────
# Fuel calculations
# ─────────────────────────────────────────────

def fuel_used(vi, vf, distance):
    """Fuel consumed over a segment with initial speed vi, final speed vf, given distance."""
    avg_speed = (vi + vf) / 2
    return (K_BASE + K_DRAG * avg_speed**2) * distance

def fuel_for_straight(entry_speed, target_speed, brake_start_m, seg_length, accel, brake_rate, max_speed):
    """
    Break the straight into 3 phases and sum fuel for each.
    Phase 1: accelerate from entry_speed to effective_target
    Phase 2: cruise at effective_target
    Phase 3: brake from effective_target to exit_speed
    """
    effective_target = min(max(target_speed, entry_speed), max_speed)
    safe_exit = None  # computed below

    L = seg_length

    # Phase 1: accelerate
    d_accel = accel_distance(entry_speed, effective_target, accel)
    if d_accel >= L:
        # Can't reach target — entire straight is acceleration
        vf_actual = math.sqrt(max(0, entry_speed**2 + 2 * accel * L))
        vf_actual = min(vf_actual, max_speed)
        return fuel_used(entry_speed, vf_actual, L), vf_actual

    # Phase 3: braking distance from target to exit
    brake_pos = L - brake_start_m  # position from start where braking begins
    d_brake_actual = L - brake_pos  # = brake_start_m

    exit_speed = math.sqrt(max(0, effective_target**2 - 2 * brake_rate * d_brake_actual))

    # Check if brake_start is before acceleration ends
    if brake_pos < d_accel:
        # Braking starts during acceleration phase
        speed_at_brake_pos = math.sqrt(max(0, entry_speed**2 + 2 * accel * brake_pos))
        speed_at_brake_pos = min(speed_at_brake_pos, max_speed)
        d_brake_actual2 = L - brake_pos
        exit_speed = math.sqrt(max(0, speed_at_brake_pos**2 - 2 * brake_rate * d_brake_actual2))
        f1 = fuel_used(entry_speed, speed_at_brake_pos, brake_pos)
        f2 = fuel_used(speed_at_brake_pos, exit_speed, d_brake_actual2)
        return f1 + f2, exit_speed

    d_cruise = brake_pos - d_accel

    f1 = fuel_used(entry_speed, effective_target, d_accel)
    f2 = fuel_used(effective_target, effective_target, d_cruise)
    f3 = fuel_used(effective_target, exit_speed, d_brake_actual)

    return f1 + f2 + f3, exit_speed

def fuel_for_corner(speed, length):
    """Constant speed through corner."""
    return fuel_used(speed, speed, length)


# ─────────────────────────────────────────────
# Straight simulation (time)
# ─────────────────────────────────────────────

def straight_exit_speed_and_time(seg, entry_speed, target_speed, brake_start_m,
                                  accel, brake_rate, max_speed):
    L = seg["length_m"]
    effective_target = min(max(target_speed, entry_speed), max_speed)

    d_accel = accel_distance(entry_speed, effective_target, accel)
    if d_accel >= L:
        vf = math.sqrt(max(0, entry_speed**2 + 2 * accel * L))
        vf = min(vf, max_speed)
        t = accel_time(entry_speed, vf, accel)
        return vf, t

    brake_pos = L - brake_start_m
    d_brake_actual = brake_start_m

    if brake_pos < d_accel:
        speed_at_brake_pos = math.sqrt(max(0, entry_speed**2 + 2 * accel * brake_pos))
        speed_at_brake_pos = min(speed_at_brake_pos, max_speed)
        exit_speed = math.sqrt(max(0, speed_at_brake_pos**2 - 2 * brake_rate * (L - brake_pos)))
        t = accel_time(entry_speed, speed_at_brake_pos, accel) + accel_time(speed_at_brake_pos, exit_speed, brake_rate)
        return exit_speed, t

    d_cruise = brake_pos - d_accel
    exit_speed = math.sqrt(max(0, effective_target**2 - 2 * brake_rate * d_brake_actual))

    t_accel  = accel_time(entry_speed, effective_target, accel)
    t_cruise = time_at_constant(d_cruise, effective_target)
    t_brake  = accel_time(exit_speed, effective_target, brake_rate)

    return exit_speed, t_accel + t_cruise + t_brake


# ─────────────────────────────────────────────
# Full race simulation with fuel tracking
# ─────────────────────────────────────────────

def simulate_race(level, strategy, verbose=False):
    car        = level["car"]
    accel      = car["accel_m/se2"]
    brake_rate = car["brake_m/se2"]
    max_speed  = car["max_speed_m/s"]
    crawl_speed = car["crawl_constant_m/s"]
    limp_speed  = car["limp_constant_m/s"]
    crash_penalty = level["race"]["corner_crash_penalty_s"]
    pit_exit_speed = level["race"]["pit_exit_speed_m/s"]
    tank_capacity  = car["fuel_tank_capacity_l"]

    compound  = get_compound(level, strategy["initial_tyre_id"])
    weather   = get_weather_at_time(level, 0)
    tyre_props = level["tyres"]["properties"][compound]
    friction  = tyre_friction(compound, weather, tyre_props, 0.0)

    segments = level["track"]["segments"]
    seg_map  = {s["id"]: s for s in segments}

    total_time   = 0.0
    current_speed = 0.0
    fuel_remaining = car["initial_fuel_l"]
    total_fuel_used = 0.0
    crashes = 0
    in_limp  = False
    in_crawl = False

    for lap_data in strategy["laps"]:
        lap_segs = lap_data["segments"]
        if verbose:
            print(f"\n--- Lap {lap_data['lap']} | Speed={current_speed:.1f} Fuel={fuel_remaining:.3f}L ---")

        for i, seg_action in enumerate(lap_segs):
            seg = seg_map[seg_action["id"]]

            if in_limp:
                # Travel entire segment at limp speed
                t = time_at_constant(seg["length_m"], limp_speed)
                f = fuel_used(limp_speed, limp_speed, seg["length_m"])
                fuel_remaining -= f
                total_fuel_used += f
                total_time += t
                current_speed = limp_speed
                if verbose:
                    print(f"  Seg {seg['id']} [LIMP] t={t:.2f}s fuel_used={f:.4f}L remaining={fuel_remaining:.4f}L")
                continue

            if seg["type"] == "straight":
                target     = seg_action.get("target_m/s", max_speed)
                brake_start = seg_action.get("brake_start_m_before_next", 0)

                exit_speed, t = straight_exit_speed_and_time(
                    seg, current_speed, target, brake_start, accel, brake_rate, max_speed)
                f, _ = fuel_for_straight(
                    current_speed, target, brake_start, seg["length_m"], accel, brake_rate, max_speed)

                fuel_remaining  -= f
                total_fuel_used += f

                if fuel_remaining < 0:
                    in_limp = True
                    if verbose:
                        print(f"  Seg {seg['id']} [straight] OUT OF FUEL -> limp mode")

                in_crawl = False
                current_speed = exit_speed
                total_time   += t

                if verbose:
                    print(f"  Seg {seg['id']} [straight] exit={exit_speed:.2f} t={t:.2f}s fuel={f:.4f}L rem={fuel_remaining:.4f}L")

            elif seg["type"] == "corner":
                entry = crawl_speed if in_crawl else current_speed
                max_c = max_corner_speed(friction, seg["radius_m"], crawl_speed)
                crashed = entry > max_c + 1e-6

                if crashed:
                    crashes += 1
                    in_crawl = True
                    t = time_at_constant(seg["length_m"], crawl_speed) + crash_penalty
                    f = fuel_for_corner(crawl_speed, seg["length_m"])
                    current_speed = crawl_speed
                else:
                    in_crawl = False
                    t = time_at_constant(seg["length_m"], entry)
                    f = fuel_for_corner(entry, seg["length_m"])
                    current_speed = entry

                fuel_remaining  -= f
                total_fuel_used += f
                if fuel_remaining < 0 and not in_limp:
                    in_limp = True

                total_time += t

                if verbose:
                    status = "CRASH" if crashed else "OK"
                    print(f"  Seg {seg['id']} [corner r={seg['radius_m']}] entry={entry:.2f} max={max_c:.2f} {status} t={t:.2f}s fuel={f:.4f}L rem={fuel_remaining:.4f}L")

        # Pit stop
        pit = lap_data.get("pit", {})
        if pit.get("enter", False):
            pit_time = level["race"]["base_pit_stop_time_s"]
            if pit.get("tyre_change_set_id"):
                pit_time += level["race"]["pit_tyre_swap_time_s"]
                new_id = pit["tyre_change_set_id"]
                compound   = get_compound(level, new_id)
                tyre_props = level["tyres"]["properties"][compound]
                friction   = tyre_friction(compound, weather, tyre_props, 0.0)

            refuel = pit.get("fuel_refuel_amount_l", 0)
            if refuel > 0:
                pit_time += refuel / level["race"]["pit_refuel_rate_l/s"]
                fuel_remaining = min(fuel_remaining + refuel, tank_capacity)

            in_limp = False  # pit fixes limp mode
            total_time += pit_time
            current_speed = pit_exit_speed

            if verbose:
                print(f"  PIT: refuel={refuel:.1f}L tyre_change={bool(pit.get('tyre_change_set_id'))} pit_time={pit_time:.1f}s fuel_now={fuel_remaining:.4f}L")

    return total_time, total_fuel_used, crashes


# ─────────────────────────────────────────────
# Fuel estimator (no simulation, fast)
# ─────────────────────────────────────────────

def estimate_total_fuel(level, strategy):
    """Quick fuel estimate without full simulation."""
    _, fuel_used_total, _ = simulate_race(level, strategy)
    return fuel_used_total


# ─────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────

def score_level2(level, total_time, total_fuel_used):
    ref = level["race"]["time_reference_s"]
    cap = level["race"]["fuel_soft_cap_limit_l"]

    base  = 500_000 * (ref / total_time) ** 3
    bonus = -500_000 * (1 - total_fuel_used / cap) ** 2 + 500_000
    return base, bonus, base + bonus


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_compound(level, tyre_id):
    for tyre_set in level["tyres"]["available_sets"]:
        if tyre_id in tyre_set["ids"]:
            return tyre_set["compound"]
    raise ValueError(f"Tyre ID {tyre_id} not found")

def get_weather_at_time(level, race_time):
    conditions = level["weather"]["conditions"]
    start_id   = level["race"].get("starting_weather_condition_id", 1)
    start_idx  = next((i for i, c in enumerate(conditions) if c["id"] == start_id), 0)
    t, idx = 0.0, start_idx
    while True:
        c = conditions[idx % len(conditions)]
        if t + c["duration_s"] > race_time:
            return c["condition"]
        t   += c["duration_s"]
        idx += 1

def compute_optimal_brake_start(seg, entry_speed, target_speed, corner_entry_speed,
                                 accel, brake_rate, max_speed):
    effective_target = min(max(target_speed, entry_speed), max_speed)
    safe_corner_speed = max(0, corner_entry_speed - 0.001)
    d_brake = accel_distance(safe_corner_speed, effective_target, brake_rate)
    return min(d_brake, seg["length_m"])

def chain_min_corner_speed(segments, start_idx, corner_max):
    """Min max-speed across consecutive corners starting at start_idx."""
    speeds = []
    j = start_idx
    while j < len(segments) and segments[j]["type"] == "corner":
        speeds.append(corner_max[segments[j]["id"]])
        j += 1
    return min(speeds) if speeds else 0.0


# ─────────────────────────────────────────────
# Strategy builder
# ─────────────────────────────────────────────

def build_lap_segments(segments, corner_max, car, target_speed_override=None):
    """
    Build the segment actions for one lap.
    target_speed_override: if set, use this instead of max_speed for all straights.
    """
    accel      = car["accel_m/se2"]
    brake_rate = car["brake_m/se2"]
    max_speed  = car["max_speed_m/s"]
    crawl      = car["crawl_constant_m/s"]

    lap_segments = []
    for i, seg in enumerate(segments):
        if seg["type"] == "straight":
            t_speed = target_speed_override if target_speed_override else max_speed

            # Required exit speed = min of chain of next corners
            chain_speed = chain_min_corner_speed(segments, i + 1, corner_max)

            brake_start = compute_optimal_brake_start(
                seg, 0, t_speed, chain_speed, accel, brake_rate, max_speed)

            lap_segments.append({
                "id": seg["id"],
                "type": "straight",
                "target_m/s": t_speed,
                "brake_start_m_before_next": round(brake_start, 4)
            })
        else:
            lap_segments.append({"id": seg["id"], "type": "corner"})

    return lap_segments


def build_strategy(level, pit_laps, refuel_amounts, target_speed=None):
    """
    Build a complete race strategy.
    pit_laps: list of lap numbers after which to pit (e.g. [1] = pit after lap 1)
    refuel_amounts: dict {lap_num: litres}
    target_speed: override speed for straights (None = max)
    """
    car      = level["car"]
    segments = level["track"]["segments"]
    num_laps = level["race"]["laps"]
    weather  = get_weather_at_time(level, 0)

    # Pick best tyre
    friction_key = WEATHER_KEY_MAP[weather][0]
    best_id, best_compound, best_friction = None, None, -1
    for tyre_set in level["tyres"]["available_sets"]:
        compound = tyre_set["compound"]
        props    = level["tyres"]["properties"][compound]
        f = BASE_FRICTION[compound] * props[friction_key]
        if f > best_friction:
            best_friction, best_compound = f, compound
            best_id = tyre_set["ids"][0]

    crawl = car["crawl_constant_m/s"]
    corner_max = {}
    for seg in segments:
        if seg["type"] == "corner":
            corner_max[seg["id"]] = max_corner_speed(best_friction, seg["radius_m"], crawl)

    lap_segs = build_lap_segments(segments, corner_max, car, target_speed)

    laps = []
    for lap_num in range(1, num_laps + 1):
        pit_enter = lap_num in pit_laps
        pit_info  = {"enter": pit_enter}
        if pit_enter:
            refuel = refuel_amounts.get(lap_num, 0)
            pit_info["fuel_refuel_amount_l"] = round(refuel, 4)
        laps.append({
            "lap": lap_num,
            "segments": deepcopy(lap_segs),
            "pit": pit_info
        })

    return {"initial_tyre_id": best_id, "laps": laps}


# ─────────────────────────────────────────────
# Fuel optimizer
# ─────────────────────────────────────────────

def estimate_fuel_per_lap_at_speed(level, target_speed):
    """Estimate fuel used per lap at a given target speed (no simulation, formula-based)."""
    car      = level["car"]
    segments = level["track"]["segments"]
    accel    = car["accel_m/se2"]
    brake_r  = car["brake_m/se2"]
    max_spd  = car["max_speed_m/s"]
    crawl    = car["crawl_constant_m/s"]

    weather       = get_weather_at_time(level, 0)
    friction_key  = WEATHER_KEY_MAP[weather][0]
    best_compound = max(
        [ts["compound"] for ts in level["tyres"]["available_sets"]],
        key=lambda c: BASE_FRICTION[c] * level["tyres"]["properties"][c][friction_key]
    )
    best_friction = BASE_FRICTION[best_compound] * level["tyres"]["properties"][best_compound][friction_key]

    corner_max = {
        seg["id"]: max_corner_speed(best_friction, seg["radius_m"], crawl)
        for seg in segments if seg["type"] == "corner"
    }

    total_fuel = 0.0
    current_speed = 0.0

    for i, seg in enumerate(segments):
        if seg["type"] == "straight":
            chain_speed = chain_min_corner_speed(segments, i + 1, corner_max)
            brake_start = compute_optimal_brake_start(seg, current_speed, target_speed, chain_speed, accel, brake_r, max_spd)
            f, exit_spd = fuel_for_straight(current_speed, target_speed, brake_start, seg["length_m"], accel, brake_r, max_spd)
            total_fuel  += f
            current_speed = exit_spd
        else:
            entry = current_speed
            total_fuel   += fuel_for_corner(entry, seg["length_m"])
            # speed stays same

    return total_fuel


def optimize_strategy(level):
    """
    Find the best strategy for Level 2:
    1. Estimate fuel per lap at max speed
    2. Determine if pit stop needed (fuel runs out mid-race)
    3. Try different pit configurations and refuel amounts
    4. Optimize target speed to hit soft cap closely for max fuel bonus
    """
    car       = level["car"]
    num_laps  = level["race"]["laps"]
    soft_cap  = level["race"]["fuel_soft_cap_limit_l"]
    tank      = car["fuel_tank_capacity_l"]
    init_fuel = car["initial_fuel_l"]

    print(f"Fuel soft cap: {soft_cap:.1f}L  |  Tank: {tank:.1f}L  |  Initial: {init_fuel:.1f}L")

    # Sweep target speeds to find one that uses fuel ≈ soft_cap
    best_score  = -float('inf')
    best_strat  = None
    best_time   = None
    best_fuel   = None

    # Speed candidates: max down to crawl, step 5
    speed_candidates = list(range(int(car["max_speed_m/s"]), int(car["crawl_constant_m/s"]), -5))

    for tgt_speed in speed_candidates:
        fuel_per_lap = estimate_fuel_per_lap_at_speed(level, tgt_speed)
        total_est_fuel = fuel_per_lap * num_laps

        # Determine if we need to pit for fuel
        # Also check if we can start with enough fuel
        pit_needed = total_est_fuel > init_fuel

        # Build candidate strategies
        candidates = []

        if not pit_needed:
            # No pit stop needed
            strat = build_strategy(level, pit_laps=[], refuel_amounts={}, target_speed=tgt_speed)
            candidates.append(strat)
        else:
            # Try pitting at end of each lap except the last
            for pit_after_lap in range(1, num_laps):
                fuel_before_pit = fuel_per_lap * pit_after_lap
                if fuel_before_pit > init_fuel:
                    # Would run out before pit — skip
                    continue
                fuel_after_pit   = fuel_per_lap * (num_laps - pit_after_lap)
                refuel_needed    = max(0, fuel_after_pit - (init_fuel - fuel_before_pit))
                refuel_needed    = min(refuel_needed, tank - (init_fuel - fuel_before_pit))
                refuel_amounts   = {pit_after_lap: refuel_needed}
                strat = build_strategy(level, pit_laps=[pit_after_lap],
                                       refuel_amounts=refuel_amounts, target_speed=tgt_speed)
                candidates.append(strat)

        for strat in candidates:
            t, fuel_used_actual, crashes = simulate_race(level, strat)
            if crashes > 0:
                continue
            base, bonus, total = score_level2(level, t, fuel_used_actual)
            if total > best_score:
                best_score = total
                best_strat = strat
                best_time  = t
                best_fuel  = fuel_used_actual

        # Early exit if we found something decent near the cap
        if best_fuel is not None and abs(best_fuel - soft_cap) / soft_cap < 0.05:
            break

    return best_strat, best_time, best_fuel, best_score


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    input_file  = sys.argv[1] if len(sys.argv) > 1 else "level2.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "submission_l2.txt"

    with open(input_file) as f:
        level = json.load(f)

    print("=== Entelect Grand Prix Solver - Level 2 ===")
    print(f"Track: {level['track']['name']}")
    print(f"Laps:  {level['race']['laps']}")
    print(f"Car:   max={level['car']['max_speed_m/s']} m/s  accel={level['car']['accel_m/se2']} m/s²  brake={level['car']['brake_m/se2']} m/s²")
    print()

    strategy, total_time, total_fuel, total_score = optimize_strategy(level)

    soft_cap = level["race"]["fuel_soft_cap_limit_l"]
    base, bonus, _ = score_level2(level, total_time, total_fuel)

    print(f"\n=== Simulation Results ===")
    print(f"Total race time:  {total_time:.2f}s")
    print(f"Total fuel used:  {total_fuel:.4f}L  (soft cap: {soft_cap:.1f}L)")
    print(f"Fuel vs cap:      {total_fuel/soft_cap*100:.1f}%")
    print(f"Base score:       {base:>15,.0f}")
    print(f"Fuel bonus:       {bonus:>15,.0f}")
    print(f"Total score:      {total_score:>15,.0f}")

    print("\n=== Strategy Summary ===")
    print(f"Initial tyre: ID {strategy['initial_tyre_id']} ({get_compound(level, strategy['initial_tyre_id'])})")
    for lap in strategy["laps"]:
        print(f"\nLap {lap['lap']}:")
        for seg in lap["segments"]:
            if seg["type"] == "straight":
                print(f"  Seg {seg['id']} [straight]: target={seg['target_m/s']} m/s  brake@{seg['brake_start_m_before_next']:.1f}m before end")
            else:
                print(f"  Seg {seg['id']} [corner]")
        pit = lap["pit"]
        if pit.get("enter"):
            print(f"  PIT: refuel={pit.get('fuel_refuel_amount_l', 0):.2f}L  tyre_change={bool(pit.get('tyre_change_set_id'))}")
        else:
            print(f"  No pit stop")

    print("\n=== Verbose Simulation ===")
    simulate_race(level, strategy, verbose=True)

    output = json.dumps(strategy, indent=2)
    with open(output_file, "w") as f:
        f.write(output)
    print(f"\nSubmission written to {output_file}")


if __name__ == "__main__":
    main()
