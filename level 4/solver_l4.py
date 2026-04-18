import json
import math
import sys
from copy import deepcopy

GRAVITY    = 9.8
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


def build_weather_timeline(level):
    conditions = level["weather"]["conditions"]
    start_id   = level["race"].get("starting_weather_condition_id", 1)
    start_idx  = next((i for i, c in enumerate(conditions) if c["id"] == start_id), 0)
    timeline, t, idx = [], 0.0, start_idx
    for _ in range(len(conditions) * 40):
        c = conditions[idx % len(conditions)]
        timeline.append((t, t + c["duration_s"], c))
        t += c["duration_s"]
        idx += 1
        if t > 500_000:
            break
    return timeline

def get_weather_at(timeline, race_time):
    for start, end, cond in timeline:
        if start <= race_time < end:
            return cond
    return timeline[-1][2]

def get_effective_accel(car, w):
    return car["accel_m/se2"] * w["acceleration_multiplier"]

def get_effective_brake(car, w):
    return car["brake_m/se2"] * w["deceleration_multiplier"]


def degrade_straight(deg_rate, length):
    return deg_rate * length * K_STRAIGHT

def degrade_braking(deg_rate, v_initial, v_final):
    return ((v_initial / 100)**2 - (v_final / 100)**2) * K_BRAKING * deg_rate

def degrade_corner(deg_rate, speed, radius):
    return K_CORNER * (speed**2 / radius) * deg_rate

def get_deg_rate(compound, weather_name, tyre_props):
    _, deg_key = WEATHER_KEY_MAP[weather_name]
    return tyre_props[deg_key]

def tyre_friction_current(compound, weather_name, tyre_props, total_deg):
    fric_key, _ = WEATHER_KEY_MAP[weather_name]
    return (BASE_FRICTION[compound] - total_deg) * tyre_props[fric_key]

def max_corner_speed(friction, radius, crawl):
    return math.sqrt(max(0.0, friction * GRAVITY * radius)) + crawl


def accel_dist(vi, vf, a):
    if a <= 0:
        return float('inf')
    return abs(vf**2 - vi**2) / (2 * a)

def accel_time_fn(vi, vf, a):
    if a <= 0:
        return 0.0
    return abs(vf - vi) / a

def fuel_used_seg(vi, vf, dist):
    avg = (vi + vf) / 2.0
    return (K_BASE + K_DRAG * avg**2) * dist

def chain_min_corner_speed(segments, start_idx, corner_max_map):
    # chained corners share no braking room — must enter at the tightest limit
    speeds, j = [], start_idx
    while j < len(segments) and segments[j]["type"] == "corner":
        speeds.append(corner_max_map[segments[j]["id"]])
        j += 1
    return min(speeds) if speeds else 0.0


def simulate_straight_full(seg, entry_speed, target_speed, brake_start_m,
                            accel, brake_rate, max_speed,
                            compound, weather_name, tyre_props, current_deg):
    L = seg["length_m"]
    deg_rate = get_deg_rate(compound, weather_name, tyre_props)
    effective_target = min(max(target_speed, entry_speed), max_speed)

    d_accel = accel_dist(entry_speed, effective_target, accel)

    if d_accel >= L:
        vf = math.sqrt(max(0, entry_speed**2 + 2 * accel * L))
        vf = min(vf, max_speed)
        t  = accel_time_fn(entry_speed, vf, accel)
        f  = fuel_used_seg(entry_speed, vf, L)
        sd = degrade_straight(deg_rate, L)
        bd = 0.0
        return vf, t, f, sd, bd, current_deg + sd + bd

    brake_pos      = L - brake_start_m
    d_brake_actual = brake_start_m

    if brake_pos <= d_accel:
        # brake point lands inside the acceleration zone — skip cruise, go straight to braking
        spd_bp = math.sqrt(max(0, entry_speed**2 + 2 * accel * brake_pos))
        spd_bp = min(spd_bp, max_speed)
        exit_spd = math.sqrt(max(0, spd_bp**2 - 2 * brake_rate * (L - brake_pos)))
        t = accel_time_fn(entry_speed, spd_bp, accel) + accel_time_fn(spd_bp, exit_spd, brake_rate)
        f = fuel_used_seg(entry_speed, spd_bp, brake_pos) + fuel_used_seg(spd_bp, exit_spd, L - brake_pos)
        sd = degrade_straight(deg_rate, L)
        bd = degrade_braking(deg_rate, spd_bp, exit_spd)
        return exit_spd, t, f, sd, bd, current_deg + sd + bd

    d_cruise = brake_pos - d_accel
    exit_spd = math.sqrt(max(0, effective_target**2 - 2 * brake_rate * d_brake_actual))

    t = (accel_time_fn(entry_speed, effective_target, accel)
         + (d_cruise / effective_target if effective_target > 0 else 0)
         + accel_time_fn(exit_spd, effective_target, brake_rate))

    f = (fuel_used_seg(entry_speed, effective_target, d_accel)
         + fuel_used_seg(effective_target, effective_target, d_cruise)
         + fuel_used_seg(effective_target, exit_spd, d_brake_actual))

    sd = degrade_straight(deg_rate, L)
    bd = degrade_braking(deg_rate, effective_target, exit_spd)

    return exit_spd, t, f, sd, bd, current_deg + sd + bd


def simulate_corner_full(seg, entry_speed, compound, weather_name, tyre_props,
                          current_deg, crawl_speed, crash_penalty):
    deg_rate = get_deg_rate(compound, weather_name, tyre_props)
    friction = tyre_friction_current(compound, weather_name, tyre_props, current_deg)
    max_c    = max_corner_speed(friction, seg["radius_m"], crawl_speed)
    crashed  = entry_speed > max_c + 1e-6

    if crashed:
        t  = seg["length_m"] / crawl_speed + crash_penalty
        f  = fuel_used_seg(crawl_speed, crawl_speed, seg["length_m"])
        cd = degrade_corner(deg_rate, crawl_speed, seg["radius_m"]) + 0.1  # extra wear from the crash itself
        return crawl_speed, t, f, cd, current_deg + cd, True
    else:
        eff_speed = max(entry_speed, crawl_speed)
        t  = seg["length_m"] / eff_speed
        f  = fuel_used_seg(eff_speed, eff_speed, seg["length_m"])
        cd = degrade_corner(deg_rate, eff_speed, seg["radius_m"])
        return eff_speed, t, f, cd, current_deg + cd, False


def get_compound(level, tyre_id):
    for ts in level["tyres"]["available_sets"]:
        if tyre_id in ts["ids"]:
            return ts["compound"]
    raise ValueError(f"Tyre ID {tyre_id} not found")

def build_tyre_inventory(level):
    id_to_compound = {}
    compound_to_ids = {}
    for ts in level["tyres"]["available_sets"]:
        compound_to_ids[ts["compound"]] = list(ts["ids"])
        for tid in ts["ids"]:
            id_to_compound[tid] = ts["compound"]
    return id_to_compound, compound_to_ids


def simulate_race(level, strategy, verbose=False):
    car           = level["car"]
    max_speed     = car["max_speed_m/s"]
    crawl_speed   = car["crawl_constant_m/s"]
    limp_speed    = car["limp_constant_m/s"]
    crash_penalty = level["race"]["corner_crash_penalty_s"]
    pit_exit_spd  = level["race"]["pit_exit_speed_m/s"]
    tank_cap      = car["fuel_tank_capacity_l"]
    life_span     = None

    timeline = build_weather_timeline(level)
    segments = level["track"]["segments"]
    seg_map  = {s["id"]: s for s in segments}
    id_to_compound, _ = build_tyre_inventory(level)

    current_tyre_id  = strategy["initial_tyre_id"]
    compound         = id_to_compound[current_tyre_id]
    tyre_props       = level["tyres"]["properties"][compound]
    life_span        = tyre_props["life_span"]
    current_deg      = 0.0

    total_time       = 0.0
    current_speed    = 0.0
    fuel_remaining   = car["initial_fuel_l"]
    total_fuel_used  = 0.0
    crashes          = 0
    blowouts         = 0
    in_limp          = False
    in_crawl         = False

    tyre_deg_history = {}

    def record_deg(tid, deg):
        tyre_deg_history[tid] = tyre_deg_history.get(tid, 0.0) + deg

    for lap_data in strategy["laps"]:
        if verbose:
            w_now = get_weather_at(timeline, total_time)
            print(f"\n--- Lap {lap_data['lap']} | t={total_time:.1f}s | spd={current_speed:.1f} | "
                  f"fuel={fuel_remaining:.2f}L | weather={w_now['condition']} | "
                  f"tyre={compound}({current_tyre_id}) deg={current_deg:.4f}/{life_span:.2f} ---")

        for seg_action in lap_data["segments"]:
            seg  = seg_map[seg_action["id"]]
            wcond = get_weather_at(timeline, total_time)
            wname = wcond["condition"]
            accel      = get_effective_accel(car, wcond)
            brake_rate = get_effective_brake(car, wcond)

            if in_limp:
                t = seg["length_m"] / limp_speed
                f = fuel_used_seg(limp_speed, limp_speed, seg["length_m"])
                # tyres still wear in limp mode, just slower
                dr = get_deg_rate(compound, wname, tyre_props)
                d  = degrade_straight(dr, seg["length_m"])
                current_deg += d
                record_deg(current_tyre_id, d)
                total_time += t; fuel_remaining -= f; total_fuel_used += f
                current_speed = limp_speed
                if verbose:
                    print(f"  Seg {seg['id']} [LIMP {seg['type']}] t={t:.2f}s")
                continue

            if seg["type"] == "straight":
                target      = seg_action.get("target_m/s", max_speed)
                brake_start = seg_action.get("brake_start_m_before_next", 0.0)

                exit_spd, t, f, sd, bd, new_deg = simulate_straight_full(
                    seg, current_speed, target, brake_start,
                    accel, brake_rate, max_speed,
                    compound, wname, tyre_props, current_deg)

                seg_deg = sd + bd
                record_deg(current_tyre_id, seg_deg)
                current_deg   = new_deg
                total_time   += t
                fuel_remaining -= f
                total_fuel_used += f
                current_speed = exit_spd
                in_crawl = False

                if current_deg >= life_span and not in_limp:
                    in_limp = True
                    blowouts += 1
                    if verbose:
                        print(f"  *** BLOWOUT on straight! deg={current_deg:.4f} ***")

                if fuel_remaining < 0 and not in_limp:
                    in_limp = True
                    if verbose:
                        print(f"  *** OUT OF FUEL on straight! ***")

                if verbose:
                    print(f"  Seg {seg['id']} [straight] w={wname} exit={exit_spd:.2f} "
                          f"t={t:.2f}s f={f:.4f}L deg+={seg_deg:.5f} total_deg={current_deg:.4f}")

            elif seg["type"] == "corner":
                entry = crawl_speed if in_crawl else current_speed
                exit_spd, t, f, cd, new_deg, crashed = simulate_corner_full(
                    seg, entry, compound, wname, tyre_props,
                    current_deg, crawl_speed, crash_penalty)

                record_deg(current_tyre_id, cd)
                current_deg    = new_deg
                total_time    += t
                fuel_remaining -= f
                total_fuel_used += f
                current_speed  = exit_spd

                if crashed:
                    crashes  += 1
                    in_crawl  = True
                else:
                    in_crawl = False

                if current_deg >= life_span and not in_limp:
                    in_limp = True
                    blowouts += 1
                    if verbose:
                        print(f"  *** BLOWOUT on corner! deg={current_deg:.4f} ***")

                if fuel_remaining < 0 and not in_limp:
                    in_limp = True

                if verbose:
                    friction = tyre_friction_current(compound, wname, tyre_props, current_deg - cd)
                    max_c = max_corner_speed(friction, seg["radius_m"], crawl_speed)
                    status = "CRASH" if crashed else "OK"
                    print(f"  Seg {seg['id']} [corner r={seg['radius_m']}] w={wname} "
                          f"entry={entry:.2f} max={max_c:.2f} {status} "
                          f"deg+={cd:.5f} total_deg={current_deg:.4f}")

        pit = lap_data.get("pit", {})
        if pit.get("enter", False):
            pit_time = level["race"]["base_pit_stop_time_s"]

            new_tyre_id = pit.get("tyre_change_set_id")
            if new_tyre_id:
                pit_time += level["race"]["pit_tyre_swap_time_s"]
                current_tyre_id = new_tyre_id
                compound        = id_to_compound[new_tyre_id]
                tyre_props      = level["tyres"]["properties"][compound]
                life_span       = tyre_props["life_span"]
                current_deg     = 0.0

            refuel = pit.get("fuel_refuel_amount_l", 0)
            if refuel > 0:
                pit_time      += refuel / level["race"]["pit_refuel_rate_l/s"]
                fuel_remaining = min(fuel_remaining + refuel, tank_cap)

            in_limp       = False  # pitting resets limp mode
            in_crawl      = False
            total_time   += pit_time
            current_speed = pit_exit_spd

            if verbose:
                c_str = f"→{compound}({new_tyre_id})" if new_tyre_id else "no change"
                print(f"  PIT: tyre {c_str}  refuel={refuel:.1f}L  pit_time={pit_time:.1f}s  fuel={fuel_remaining:.2f}L")

    total_tyre_degradation = sum(tyre_deg_history.values())

    return total_time, total_fuel_used, crashes, blowouts, total_tyre_degradation, tyre_deg_history


def score_level4(level, total_time, total_fuel, blowouts, total_tyre_deg):
    ref = level["race"]["time_reference_s"]
    cap = level["race"]["fuel_soft_cap_limit_l"]

    base        = 500_000 * (ref / total_time) ** 3
    fuel_bonus  = -500_000 * (1 - total_fuel / cap) ** 2 + 500_000
    tyre_bonus  = 100_000 * total_tyre_deg - 50_000 * blowouts
    total       = base + fuel_bonus + tyre_bonus
    return base, fuel_bonus, tyre_bonus, total


def estimate_deg_per_lap(compound, weather_name, level, target_speed, friction, crawl):
    tyre_props = level["tyres"]["properties"][compound]
    deg_rate   = get_deg_rate(compound, weather_name, tyre_props)
    segments   = level["track"]["segments"]
    car        = level["car"]
    max_speed  = car["max_speed_m/s"]
    accel      = car["accel_m/se2"]
    brake_rate = car["brake_m/se2"]

    corner_max = {
        seg["id"]: max_corner_speed(friction, seg["radius_m"], crawl)
        for seg in segments if seg["type"] == "corner"
    }

    total_deg = 0.0
    for i, seg in enumerate(segments):
        if seg["type"] == "straight":
            chain_spd = chain_min_corner_speed(segments, i + 1, corner_max)
            total_deg += degrade_straight(deg_rate, seg["length_m"])
            eff_target = min(target_speed, max_speed)
            total_deg += degrade_braking(deg_rate, eff_target, max(chain_spd, 0))
        else:
            speeds = []
            j = i
            while j < len(segments) and segments[j]["type"] == "corner":
                speeds.append(corner_max[segments[j]["id"]])
                j += 1
            spd = min(speeds) if speeds else crawl
            total_deg += degrade_corner(deg_rate, spd, seg["radius_m"])

    return total_deg


def build_lap_segs(segments, level, compound, weather_name, car,
                   current_deg, timeline=None, lap_start_time=0.0):
    tyre_props = level["tyres"]["properties"][compound]
    max_speed  = car["max_speed_m/s"]
    crawl      = car["crawl_constant_m/s"]

    all_conds_list = level["weather"]["conditions"]
    all_conds  = {c["condition"]: c for c in all_conds_list}
    w_cond     = all_conds.get(weather_name,
                               {"acceleration_multiplier": 1.0, "deceleration_multiplier": 1.0})
    accel      = car["accel_m/se2"]  * w_cond["acceleration_multiplier"]
    brake_rate = car["brake_m/se2"]  * w_cond["deceleration_multiplier"]

    friction = tyre_friction_current(compound, weather_name, tyre_props, current_deg)

    # Use worst-case friction across weathers that could hit this lap for corner limits —
    # this keeps brake distances safe even if conditions worsen mid-lap.
    # brake_rate from dominant weather must match what the simulator will actually apply.
    worst_friction = friction
    if timeline:
        for start, end, cond in timeline:
            if start > lap_start_time + 400:
                break
            if end < lap_start_time:
                continue
            f = tyre_friction_current(compound, cond["condition"], tyre_props, current_deg)
            worst_friction = min(worst_friction, f)

    corner_max = {
        seg["id"]: max_corner_speed(worst_friction, seg["radius_m"], crawl)
        for seg in segments if seg["type"] == "corner"
    }

    lap_segs = []
    for i, seg in enumerate(segments):
        if seg["type"] == "straight":
            chain_spd   = chain_min_corner_speed(segments, i + 1, corner_max)
            safe_chain  = max(chain_spd - 0.001, crawl)
            d_brake     = accel_dist(safe_chain, car["max_speed_m/s"], brake_rate)
            brake_start = min(d_brake, seg["length_m"])
            lap_segs.append({
                "id":   seg["id"],
                "type": "straight",
                "target_m/s": max_speed,
                "brake_start_m_before_next": round(brake_start, 4)
            })
        else:
            lap_segs.append({"id": seg["id"], "type": "corner"})

    return lap_segs, corner_max, friction


def plan_pit_strategy(level, timeline):
    car       = level["car"]
    segments  = level["track"]["segments"]
    num_laps  = level["race"]["laps"]
    crawl     = car["crawl_constant_m/s"]
    max_speed = car["max_speed_m/s"]

    id_to_compound, compound_to_ids = build_tyre_inventory(level)
    used_ids = set()

    track_length = sum(s["length_m"] for s in segments)
    rough_lap_t  = track_length / 45.0

    w0_name   = get_weather_at(timeline, 0.0)["condition"]
    init_c, init_id = pick_best_tyre(level, w0_name, used_ids, compound_to_ids, id_to_compound)
    used_ids.add(init_id)

    print(f"Track length: {track_length}m  Rough lap time: {rough_lap_t:.1f}s")
    print(f"Initial tyre: {init_c} (ID {init_id})")
    print()

    plan = []
    current_deg  = 0.0
    current_id   = init_id
    current_c    = init_c
    est_race_time = 0.0
    fuel_remaining = car["initial_fuel_l"]
    fuel_per_lap_est = track_length * K_BASE * 1.15

    for lap in range(1, num_laps + 1):
        w_cond  = get_weather_at(timeline, est_race_time)
        w_name  = w_cond["condition"]

        tyre_props = level["tyres"]["properties"][current_c]
        friction   = tyre_friction_current(current_c, w_name, tyre_props, current_deg)
        life_span  = tyre_props["life_span"]

        deg_this_lap = estimate_deg_per_lap(current_c, w_name, level, max_speed, friction, crawl)
        deg_after = current_deg + deg_this_lap

        next_w_name = get_weather_at(timeline, est_race_time + rough_lap_t)["condition"] if lap < num_laps else None
        next_best_c = None
        if next_w_name:
            next_best_c, _ = pick_best_tyre(level, next_w_name, set(), compound_to_ids, id_to_compound)

        safety_margin = 0.05  # keep 5% tyre life in reserve to avoid surprise blowouts
        next_deg_est = 0.0
        if lap < num_laps and next_w_name:
            next_friction = tyre_friction_current(current_c, next_w_name, tyre_props, deg_after)
            next_deg_est  = estimate_deg_per_lap(current_c, next_w_name, level, max_speed, next_friction, crawl)

        will_blowout_next = (deg_after + next_deg_est) >= (life_span - safety_margin)
        compound_change   = next_best_c != current_c if next_best_c else False
        need_fuel_pit     = fuel_remaining - fuel_per_lap_est < fuel_per_lap_est and lap < num_laps

        plan.append({
            "lap":        lap,
            "tyre_id":    current_id,
            "compound":   current_c,
            "weather":    w_name,
            "deg_start":  current_deg,
            "deg_end":    deg_after,
            "life_span":  life_span,
        })

        current_deg = deg_after
        est_race_time += rough_lap_t
        fuel_remaining -= fuel_per_lap_est

        should_pit = (will_blowout_next or compound_change or need_fuel_pit) and lap < num_laps

        if should_pit:
            if next_w_name:
                new_c, new_id = pick_best_tyre(level, next_w_name, used_ids, compound_to_ids, id_to_compound)
            else:
                new_c, new_id = pick_best_tyre(level, w_name, used_ids, compound_to_ids, id_to_compound)

            if new_id is None:
                new_c, new_id = pick_best_tyre(level, next_w_name or w_name,
                                                set(), compound_to_ids, id_to_compound)

            reason = []
            if will_blowout_next: reason.append(f"blowout risk (deg={deg_after:.3f}+{next_deg_est:.3f}>={life_span-safety_margin:.2f})")
            if compound_change:   reason.append(f"weather change {w_name}→{next_w_name} prefers {next_best_c}")
            if need_fuel_pit:     reason.append("low fuel")

            print(f"  Pit after lap {lap}: {current_c}→{new_c}(ID {new_id}) | {', '.join(reason)}")
            plan[-1]["pit_after"] = True
            plan[-1]["new_tyre_id"] = new_id
            plan[-1]["new_compound"] = new_c

            used_ids.add(new_id)
            current_id  = new_id
            current_c   = new_c
            current_deg = 0.0

            fuel_needed_remaining = fuel_per_lap_est * (num_laps - lap)
            refuel = max(0.0, fuel_needed_remaining - max(fuel_remaining, 0))
            refuel = min(refuel, car["fuel_tank_capacity_l"] - max(fuel_remaining, 0))
            plan[-1]["refuel"] = round(refuel, 4)
            if refuel > 0:
                fuel_remaining += refuel
                print(f"    Also refuel: {refuel:.2f}L")
        else:
            plan[-1]["pit_after"] = False
            plan[-1]["new_tyre_id"] = None
            plan[-1]["refuel"] = 0.0

    return plan, init_id, init_c


def pick_best_tyre(level, weather_name, used_ids, compound_to_ids, id_to_compound):
    friction_key = WEATHER_KEY_MAP[weather_name][0]
    best_id, best_c, best_f = None, None, -1

    for compound, ids in compound_to_ids.items():
        props = level["tyres"]["properties"][compound]
        f = BASE_FRICTION[compound] * props[friction_key]
        available = [i for i in ids if i not in used_ids]
        if available and f > best_f:
            best_f = f
            best_c = compound
            best_id = available[0]

    return best_c, best_id


def build_strategy(level, plan, timeline):
    car      = level["car"]
    segments = level["track"]["segments"]

    id_to_compound, _ = build_tyre_inventory(level)
    laps_output = []

    for entry in plan:
        lap_num    = entry["lap"]
        compound   = entry["compound"]
        weather    = entry["weather"]
        deg_start  = entry["deg_start"]
        est_t_start = (lap_num - 1) * sum(s["length_m"] for s in segments) / 45.0

        lap_segs, _, _ = build_lap_segs(
            segments, level, compound, weather, car,
            deg_start, timeline=timeline, lap_start_time=est_t_start)

        pit_info = {"enter": False}
        if entry["pit_after"]:
            pit_info = {"enter": True}
            if entry["new_tyre_id"]:
                pit_info["tyre_change_set_id"] = entry["new_tyre_id"]
            if entry["refuel"] > 0:
                pit_info["fuel_refuel_amount_l"] = entry["refuel"]

        laps_output.append({
            "lap":      lap_num,
            "segments": lap_segs,
            "pit":      pit_info
        })

    return {
        "initial_tyre_id": plan[0]["tyre_id"],
        "laps": laps_output
    }


def main():
    input_file  = sys.argv[1] if len(sys.argv) > 1 else "level4.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "submission_l4.txt"

    with open(input_file) as f:
        level = json.load(f)

    print("=== Entelect Grand Prix Solver - Level 4 ===")
    print(f"Track: {level['track']['name']}")
    print(f"Laps:  {level['race']['laps']}")
    print(f"Car:   max={level['car']['max_speed_m/s']} m/s  "
          f"accel={level['car']['accel_m/se2']} m/s²  "
          f"brake={level['car']['brake_m/se2']} m/s²")
    print()

    timeline = build_weather_timeline(level)

    print("Weather schedule (first 16 windows):")
    for start, end, cond in timeline[:16]:
        print(f"  {start:>7.1f}s – {end:>7.1f}s : {cond['condition']:12s} "
              f"(accel×{cond['acceleration_multiplier']}  brake×{cond['deceleration_multiplier']})")
    print()

    print("=== Building pit strategy ===")
    plan, init_id, init_c = plan_pit_strategy(level, timeline)

    print("\n=== Lap-by-lap degradation plan ===")
    for entry in plan:
        bar_used  = int(entry["deg_end"] / entry["life_span"] * 20)
        bar_empty = 20 - bar_used
        bar = "█" * bar_used + "░" * bar_empty
        print(f"  Lap {entry['lap']:2d} [{entry['compound']:13s}] [{bar}] "
              f"{entry['deg_end']:.4f}/{entry['life_span']:.2f} "
              f"weather={entry['weather']:10s}"
              + (f"  → PIT (new: {entry.get('new_compound','?')} ID {entry.get('new_tyre_id','?')})" if entry['pit_after'] else ""))

    strategy = build_strategy(level, plan, timeline)

    print("\n=== Running full simulation ===")
    total_time, total_fuel, crashes, blowouts, total_tyre_deg, tyre_history = \
        simulate_race(level, strategy, verbose=False)

    base, fuel_bonus, tyre_bonus, total_score = score_level4(
        level, total_time, total_fuel, blowouts, total_tyre_deg)

    soft_cap = level["race"]["fuel_soft_cap_limit_l"]
    print(f"\n=== Simulation Results ===")
    print(f"Total race time:       {total_time:.2f}s")
    print(f"Total fuel used:       {total_fuel:.4f}L  (cap={soft_cap:.1f}L, {total_fuel/soft_cap*100:.1f}%)")
    print(f"Crashes:               {crashes}")
    print(f"Blowouts:              {blowouts}")
    print(f"Total tyre degradation:{total_tyre_deg:.5f}")
    print(f"Tyre sets used:        {list(tyre_history.keys())}")
    print(f"\nBase score:            {base:>15,.0f}")
    print(f"Fuel bonus:            {fuel_bonus:>15,.0f}")
    print(f"Tyre bonus:            {tyre_bonus:>15,.0f}  (100k×{total_tyre_deg:.4f} - 50k×{blowouts})")
    print(f"Total score:           {total_score:>15,.0f}")

    print("\n=== Verbose Race Trace ===")
    simulate_race(level, strategy, verbose=True)

    with open(output_file, "w") as f:
        f.write(json.dumps(strategy, indent=2))
    print(f"\nSubmission written to {output_file}")


if __name__ == "__main__":
    main()
