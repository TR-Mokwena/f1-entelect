import json
import math
import sys
from copy import deepcopy

GRAVITY    = 9.8
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

BEST_TYRE_FOR_WEATHER = {
    "dry":        "Soft",
    "cold":       "Soft",
    "light_rain": "Intermediate",
    "heavy_rain": "Wet",
}


def build_weather_timeline(level):
    conditions = level["weather"]["conditions"]
    start_id   = level["race"].get("starting_weather_condition_id", 1)
    start_idx  = next((i for i, c in enumerate(conditions) if c["id"] == start_id), 0)

    timeline = []
    t = 0.0
    idx = start_idx
    for _ in range(len(conditions) * 20):
        c = conditions[idx % len(conditions)]
        timeline.append((t, t + c["duration_s"], c))
        t += c["duration_s"]
        idx += 1
        if t > 100_000:
            break
    return timeline


def get_weather_at(timeline, race_time):
    for start, end, cond in timeline:
        if start <= race_time < end:
            return cond
    return timeline[-1][2]


def get_effective_accel(car, weather_cond):
    return car["accel_m/se2"] * weather_cond["acceleration_multiplier"]

def get_effective_brake(car, weather_cond):
    return car["brake_m/se2"] * weather_cond["deceleration_multiplier"]

def tyre_friction_for(compound, weather_name, tyre_props, degradation=0.0):
    friction_key = WEATHER_KEY_MAP[weather_name][0]
    return (BASE_FRICTION[compound] - degradation) * tyre_props[friction_key]

def max_corner_speed(friction, radius, crawl_speed):
    return math.sqrt(max(0.0, friction * GRAVITY * radius)) + crawl_speed

def best_tyre_for_weather(weather_name, level, used_ids=None):
    friction_key = WEATHER_KEY_MAP[weather_name][0]
    best_id, best_compound, best_f = None, None, -1
    for tyre_set in level["tyres"]["available_sets"]:
        compound = tyre_set["compound"]
        props    = level["tyres"]["properties"][compound]
        f = BASE_FRICTION[compound] * props[friction_key]
        available = [i for i in tyre_set["ids"] if used_ids is None or i not in used_ids]
        if available and f > best_f:
            best_f, best_compound, best_id = f, compound, available[0]
    return best_compound, best_id, best_f


def accel_distance(vi, vf, a):
    if a == 0:
        return float('inf')
    return abs(vf**2 - vi**2) / (2 * a)

def accel_time_fn(vi, vf, a):
    if a == 0:
        return 0.0
    return abs(vf - vi) / a

def fuel_used_seg(vi, vf, distance):
    avg = (vi + vf) / 2
    return (K_BASE + K_DRAG * avg**2) * distance

def chain_min_corner_speed_fn(segments, start_idx, corner_max_map):
    # chained corners share no braking room — must enter at the tightest limit
    speeds = []
    j = start_idx
    while j < len(segments) and segments[j]["type"] == "corner":
        speeds.append(corner_max_map[segments[j]["id"]])
        j += 1
    return min(speeds) if speeds else 0.0

def compute_brake_start(seg, target_speed, corner_entry_speed, accel, brake_rate, max_speed, entry_speed=0):
    effective_target = min(max(target_speed, entry_speed), max_speed)
    safe_corner      = max(0.0, corner_entry_speed - 0.001)  # tiny margin to avoid floating-point crashes
    d_brake          = accel_distance(safe_corner, effective_target, brake_rate)
    return min(d_brake, seg["length_m"])


def simulate_straight_seg(seg, entry_speed, target_speed, brake_start_m,
                           accel, brake_rate, max_speed):
    L = seg["length_m"]
    effective_target = min(max(target_speed, entry_speed), max_speed)

    d_accel = accel_distance(entry_speed, effective_target, accel)

    if d_accel >= L:
        vf = math.sqrt(max(0, entry_speed**2 + 2 * accel * L))
        vf = min(vf, max_speed)
        t  = accel_time_fn(entry_speed, vf, accel)
        f  = fuel_used_seg(entry_speed, vf, L)
        return vf, t, f

    brake_pos     = L - brake_start_m
    d_brake_actual = brake_start_m

    if brake_pos <= d_accel:
        # brake point lands inside the acceleration zone — skip cruise, go straight to braking
        spd_at_bp = math.sqrt(max(0, entry_speed**2 + 2 * accel * brake_pos))
        spd_at_bp = min(spd_at_bp, max_speed)
        exit_spd  = math.sqrt(max(0, spd_at_bp**2 - 2 * brake_rate * (L - brake_pos)))
        t = accel_time_fn(entry_speed, spd_at_bp, accel) + accel_time_fn(spd_at_bp, exit_spd, brake_rate)
        f = fuel_used_seg(entry_speed, spd_at_bp, brake_pos) + fuel_used_seg(spd_at_bp, exit_spd, L - brake_pos)
        return exit_spd, t, f

    d_cruise  = brake_pos - d_accel
    exit_spd  = math.sqrt(max(0, effective_target**2 - 2 * brake_rate * d_brake_actual))

    t = (accel_time_fn(entry_speed, effective_target, accel)
         + (d_cruise / effective_target if effective_target > 0 else 0)
         + accel_time_fn(exit_spd, effective_target, brake_rate))

    f = (fuel_used_seg(entry_speed, effective_target, d_accel)
         + fuel_used_seg(effective_target, effective_target, d_cruise)
         + fuel_used_seg(effective_target, exit_spd, d_brake_actual))

    return exit_spd, t, f


def simulate_race(level, strategy, verbose=False):
    car           = level["car"]
    max_speed     = car["max_speed_m/s"]
    crawl_speed   = car["crawl_constant_m/s"]
    limp_speed    = car["limp_constant_m/s"]
    crash_penalty = level["race"]["corner_crash_penalty_s"]
    pit_exit_spd  = level["race"]["pit_exit_speed_m/s"]
    tank_cap      = car["fuel_tank_capacity_l"]

    timeline = build_weather_timeline(level)
    segments = level["track"]["segments"]
    seg_map  = {s["id"]: s for s in segments}

    current_tyre_id = strategy["initial_tyre_id"]
    compound        = get_compound(level, current_tyre_id)

    total_time      = 0.0
    current_speed   = 0.0
    fuel_remaining  = car["initial_fuel_l"]
    total_fuel_used = 0.0
    crashes         = 0
    in_limp         = False
    in_crawl        = False

    for lap_data in strategy["laps"]:
        if verbose:
            weather_now = get_weather_at(timeline, total_time)
            print(f"\n--- Lap {lap_data['lap']} | t={total_time:.1f}s | speed={current_speed:.1f} | "
                  f"fuel={fuel_remaining:.3f}L | weather={weather_now['condition']} | tyre={compound} ---")

        for seg_action in lap_data["segments"]:
            seg          = seg_map[seg_action["id"]]
            weather_cond = get_weather_at(timeline, total_time)
            w_name       = weather_cond["condition"]
            accel        = get_effective_accel(car, weather_cond)
            brake_rate   = get_effective_brake(car, weather_cond)

            tyre_props   = level["tyres"]["properties"][compound]
            friction     = tyre_friction_for(compound, w_name, tyre_props)

            if in_limp:
                t = seg["length_m"] / limp_speed
                f = fuel_used_seg(limp_speed, limp_speed, seg["length_m"])
                total_time += t; fuel_remaining -= f; total_fuel_used += f
                current_speed = limp_speed
                if verbose:
                    print(f"  Seg {seg['id']} [LIMP {seg['type']}] t={t:.2f}s f={f:.4f}L")
                continue

            if seg["type"] == "straight":
                target      = seg_action.get("target_m/s", max_speed)
                brake_start = seg_action.get("brake_start_m_before_next", 0.0)

                exit_spd, t, f = simulate_straight_seg(
                    seg, current_speed, target, brake_start, accel, brake_rate, max_speed)

                total_time    += t
                fuel_remaining -= f
                total_fuel_used += f
                current_speed = exit_spd
                in_crawl = False

                if fuel_remaining < 0 and not in_limp:
                    in_limp = True

                if verbose:
                    print(f"  Seg {seg['id']} [straight] weather={w_name} accel={accel:.1f} "
                          f"exit={exit_spd:.2f} t={t:.2f}s f={f:.4f}L rem={fuel_remaining:.3f}L")

            elif seg["type"] == "corner":
                entry_speed = crawl_speed if in_crawl else current_speed
                max_c       = max_corner_speed(friction, seg["radius_m"], crawl_speed)
                crashed     = entry_speed > max_c + 1e-6

                if crashed:
                    crashes  += 1
                    in_crawl  = True
                    t = seg["length_m"] / crawl_speed + crash_penalty
                    f = fuel_used_seg(crawl_speed, crawl_speed, seg["length_m"])
                    current_speed = crawl_speed
                else:
                    in_crawl = False
                    t = seg["length_m"] / entry_speed if entry_speed > 0 else float('inf')
                    f = fuel_used_seg(entry_speed, entry_speed, seg["length_m"])
                    current_speed = entry_speed

                total_time    += t
                fuel_remaining -= f
                total_fuel_used += f
                if fuel_remaining < 0 and not in_limp:
                    in_limp = True

                if verbose:
                    status = "CRASH" if crashed else "OK"
                    print(f"  Seg {seg['id']} [corner r={seg['radius_m']}] weather={w_name} "
                          f"friction={friction:.3f} entry={entry_speed:.2f} max={max_c:.2f} "
                          f"{status} t={t:.2f}s f={f:.4f}L rem={fuel_remaining:.3f}L")

        pit = lap_data.get("pit", {})
        if pit.get("enter", False):
            pit_time = level["race"]["base_pit_stop_time_s"]

            new_tyre_id = pit.get("tyre_change_set_id")
            if new_tyre_id:
                pit_time       += level["race"]["pit_tyre_swap_time_s"]
                current_tyre_id = new_tyre_id
                compound        = get_compound(level, new_tyre_id)

            refuel = pit.get("fuel_refuel_amount_l", 0)
            if refuel > 0:
                pit_time       += refuel / level["race"]["pit_refuel_rate_l/s"]
                fuel_remaining  = min(fuel_remaining + refuel, tank_cap)

            in_limp       = False  # pitting resets limp mode
            total_time   += pit_time
            current_speed = pit_exit_spd

            if verbose:
                print(f"  PIT: new_tyre={compound}({new_tyre_id}) refuel={refuel:.1f}L "
                      f"pit_time={pit_time:.1f}s fuel_now={fuel_remaining:.3f}L")

    return total_time, total_fuel_used, crashes


def score_level3(level, total_time, total_fuel):
    ref = level["race"]["time_reference_s"]
    cap = level["race"]["fuel_soft_cap_limit_l"]
    base  = 500_000 * (ref / total_time) ** 3
    bonus = -500_000 * (1 - total_fuel / cap) ** 2 + 500_000
    return base, bonus, base + bonus


def get_compound(level, tyre_id):
    for tyre_set in level["tyres"]["available_sets"]:
        if tyre_id in tyre_set["ids"]:
            return tyre_set["compound"]
    raise ValueError(f"Tyre ID {tyre_id} not found")


def build_lap_segments_for_weather(segments, level, compound, weather_name, car,
                                    entry_speed=0.0, timeline=None, lap_start_time=0.0):
    all_conds_list = level["weather"]["conditions"]
    max_speed = car["max_speed_m/s"]
    crawl     = car["crawl_constant_m/s"]

    tyre_props = level["tyres"]["properties"][compound]
    friction   = tyre_friction_for(compound, weather_name, tyre_props)

    dom_cond = next((c for c in all_conds_list if c["condition"] == weather_name),
                    {"acceleration_multiplier": 1.0, "deceleration_multiplier": 1.0})

    accel      = car["accel_m/se2"]  * dom_cond["acceleration_multiplier"]
    brake_rate = car["brake_m/se2"]  * dom_cond["deceleration_multiplier"]

    # Use the worst friction across all weathers that could hit during this lap —
    # conservative corner speed limits keep us safe if conditions worsen mid-lap
    worst_friction = friction
    if timeline is not None:
        for start, end, cond in timeline:
            if start > lap_start_time + 400:
                break
            if end < lap_start_time:
                continue
            f = tyre_friction_for(compound, cond["condition"], tyre_props)
            worst_friction = min(worst_friction, f)

    corner_max = {}
    for seg in segments:
        if seg["type"] == "corner":
            corner_max[seg["id"]] = max_corner_speed(worst_friction, seg["radius_m"], crawl)

    lap_segs = []

    for i, seg in enumerate(segments):
        if seg["type"] == "straight":
            chain_spd   = chain_min_corner_speed_fn(segments, i + 1, corner_max)
            brake_start = compute_brake_start(
                seg, max_speed, chain_spd, accel, brake_rate, max_speed, entry_speed)
            lap_segs.append({
                "id":   seg["id"],
                "type": "straight",
                "target_m/s": max_speed,
                "brake_start_m_before_next": round(brake_start, 4)
            })
            entry_speed = max(chain_spd, 0.0)
        else:
            lap_segs.append({"id": seg["id"], "type": "corner"})

    return lap_segs, corner_max


def predict_lap_weather(level, timeline, lap_num, lap_time_estimate):
    start_t = (lap_num - 1) * lap_time_estimate
    mid_t = start_t + lap_time_estimate / 2
    cond = get_weather_at(timeline, mid_t)
    return cond["condition"]


def build_optimal_strategy(level):
    car      = level["car"]
    segments = level["track"]["segments"]
    num_laps = level["race"]["laps"]
    timeline = build_weather_timeline(level)

    track_length = sum(s["length_m"] for s in segments)
    rough_lap_time = track_length / 50.0

    print(f"Track length: {track_length}m  |  Rough lap time estimate: {rough_lap_time:.1f}s")
    print(f"Number of laps: {num_laps}")
    print()

    print("Weather forecast per lap:")
    lap_weathers = []
    for lap in range(1, num_laps + 1):
        w = predict_lap_weather(level, timeline, lap, rough_lap_time)
        lap_weathers.append(w)
        best_c, best_id, best_f = best_tyre_for_weather(w, level)
        print(f"  Lap {lap}: {w:12s} → best tyre: {best_c} (friction={best_f:.4f})")
    print()

    used_tyre_ids = set()

    init_compound, init_tyre_id, _ = best_tyre_for_weather(lap_weathers[0], level, used_tyre_ids)
    used_tyre_ids.add(init_tyre_id)
    current_compound = init_compound

    pit_plan = {}
    fuel = car["initial_fuel_l"]
    fuel_per_lap_est = track_length * K_BASE * 1.1

    for lap in range(1, num_laps + 1):
        fuel -= fuel_per_lap_est

        if lap < num_laps:
            next_weather    = lap_weathers[lap]  # lap is 1-indexed, so lap_weathers[lap] is next lap
            best_next_c, best_next_id, _ = best_tyre_for_weather(next_weather, level, used_tyre_ids)
            tyre_change_needed = best_next_c != current_compound

            fuel_needed_next = fuel_per_lap_est * (num_laps - lap)
            refuel_needed    = max(0.0, fuel_needed_next - fuel)
            refuel_needed    = min(refuel_needed, car["fuel_tank_capacity_l"] - max(0, fuel))

            if tyre_change_needed or refuel_needed > 0:
                pit_plan[lap] = {}
                if tyre_change_needed:
                    pit_plan[lap]["tyre_change_set_id"]  = best_next_id
                    pit_plan[lap]["compound"]            = best_next_c
                    used_tyre_ids.add(best_next_id)
                    current_compound = best_next_c
                    print(f"  Pit after lap {lap}: tyre change → {best_next_c} (ID {best_next_id}) due to weather: {next_weather}")
                if refuel_needed > 0:
                    pit_plan[lap]["fuel_refuel_amount_l"] = round(refuel_needed, 4)
                    fuel += refuel_needed
                    print(f"  Pit after lap {lap}: refuel {refuel_needed:.2f}L")

    laps_output = []
    current_compound = init_compound
    estimated_lap_time = rough_lap_time
    estimated_race_time = 0.0

    for lap_num in range(1, num_laps + 1):
        w_name   = lap_weathers[lap_num - 1]

        lap_segs, _ = build_lap_segments_for_weather(
            segments, level, current_compound, w_name, car,
            timeline=timeline, lap_start_time=estimated_race_time)

        estimated_race_time += estimated_lap_time

        pit_info = {"enter": False}
        if lap_num in pit_plan:
            plan = pit_plan[lap_num]
            pit_info = {"enter": True}
            if "tyre_change_set_id" in plan:
                pit_info["tyre_change_set_id"] = plan["tyre_change_set_id"]
            if "fuel_refuel_amount_l" in plan:
                pit_info["fuel_refuel_amount_l"] = plan["fuel_refuel_amount_l"]
            if "compound" in plan:
                current_compound = plan["compound"]

        laps_output.append({
            "lap":      lap_num,
            "segments": lap_segs,
            "pit":      pit_info
        })

    return {
        "initial_tyre_id": init_tyre_id,
        "laps":            laps_output
    }


def main():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, "level3.json")
    output_file = sys.argv[2] if len(sys.argv) > 2 else os.path.join(script_dir, "3.txt")

    with open(input_file) as f:
        level = json.load(f)

    print("=== Entelect Grand Prix Solver - Level 3 ===")
    print(f"Track: {level['track']['name']}")
    print(f"Laps:  {level['race']['laps']}")
    print(f"Car:   max={level['car']['max_speed_m/s']} m/s  "
          f"accel={level['car']['accel_m/se2']} m/s²  "
          f"brake={level['car']['brake_m/se2']} m/s²")
    print()

    timeline = build_weather_timeline(level)
    print("Weather schedule:")
    for start, end, cond in timeline[:12]:
        print(f"  {start:>7.1f}s – {end:>7.1f}s : {cond['condition']:12s} "
              f"(accel×{cond['acceleration_multiplier']}  brake×{cond['deceleration_multiplier']})")
    print()

    strategy = build_optimal_strategy(level)

    total_time, total_fuel, crashes = simulate_race(level, strategy, verbose=False)
    base, bonus, total = score_level3(level, total_time, total_fuel)
    soft_cap = level["race"]["fuel_soft_cap_limit_l"]

    print(f"\n=== Simulation Results ===")
    print(f"Total race time:  {total_time:.2f}s")
    print(f"Total fuel used:  {total_fuel:.4f}L  (soft cap: {soft_cap:.1f}L, {total_fuel/soft_cap*100:.1f}%)")
    print(f"Crashes:          {crashes}")
    print(f"Base score:       {base:>15,.0f}")
    print(f"Fuel bonus:       {bonus:>15,.0f}")
    print(f"Total score:      {total:>15,.0f}")

    print("\n=== Verbose Race Trace ===")
    simulate_race(level, strategy, verbose=True)

    print("\n=== Strategy Summary ===")
    print(f"Initial tyre: ID {strategy['initial_tyre_id']} ({get_compound(level, strategy['initial_tyre_id'])})")
    for lap in strategy["laps"]:
        print(f"\nLap {lap['lap']}:")
        for seg in lap["segments"]:
            if seg["type"] == "straight":
                print(f"  Seg {seg['id']} [straight]: target={seg['target_m/s']} m/s  "
                      f"brake@{seg['brake_start_m_before_next']:.1f}m before end")
            else:
                print(f"  Seg {seg['id']} [corner]")
        pit = lap["pit"]
        if pit.get("enter"):
            tc = pit.get("tyre_change_set_id", "none")
            rf = pit.get("fuel_refuel_amount_l", 0)
            print(f"  PIT: tyre→{tc}  refuel={rf:.2f}L")
        else:
            print("  No pit stop")

    with open(output_file, "w") as f:
        f.write(json.dumps(strategy, indent=2))
    print(f"\nSubmission written to {output_file}")


if __name__ == "__main__":
    main()
