"""
M8 — Event Impact Simulator
Planned-event traffic surge modelling focusing primarily on construction zone traffic
planning (the dominant planned event type in Astram), and secondarily on public gatherings,
processions, VIP movements, and protests.

Calibrated from Astram dataset planned-event signatures (construction, public_event, procession,
vip_movement) — multipliers anchored to observed congestion footprints.

Outputs: models/event_sim_results.json  (per-simulation)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

DATA_DIR  = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"
FIG_DIR   = Path(__file__).parent.parent / "figures"
MODEL_DIR.mkdir(exist_ok=True)

# ─── Venue → primary corridor mapping ────────────────────────────────────────
VENUE_CORRIDORS: Dict[str, List[str]] = {
    "Chinnaswamy Stadium":       ["Bellary Road 1", "Old Madras Road"],
    "Kanteerava Stadium":        ["Bellary Road 1", "Hosur Road"],
    "KSCA Cricket Ground":       ["Bellary Road 1", "Bellary Road 2"],
    "Lalbagh Botanical Garden":  ["Hosur Road", "Bannerghatta Road"],
    "Cubbon Park":               ["Old Madras Road", "West of Chord Road"],
    "Palace Grounds":            ["Bellary Road 1", "Tumkur Road"],
    "Freedom Park":              ["West of Chord Road", "Magadi Road"],
    "Vidhana Soudha":            ["West of Chord Road", "Bellary Road 1"],
    "Forum Mall (Koramangala)":  ["Hosur Road", "ORR East 1"],
    "Orion Mall (Rajajinagar)":  ["Tumkur Road", "West of Chord Road"],
    "Custom Venue":              ["Mysore Road"],
}

def _hhmm(h: float) -> str:
    hh = int(h) % 24
    mm = int((h % 1) * 60)
    return f"{hh:02d}:{mm:02d}"

# ─── Crowd → surge multiplier (events per km·hour relative to baseline) ──────
# Anchored to Astram: planned events create 1.8–4.2× more congestion events
# compared to same hour on non-event days.
def crowd_to_surge(crowd_size: int) -> float:
    """Piecewise-linear crowd-to-surge conversion."""
    if crowd_size <= 0:       return 1.0
    if crowd_size < 1_000:    return 1.3
    if crowd_size < 5_000:    return 1.5 + 0.5 * (crowd_size - 1_000) / 4_000
    if crowd_size < 20_000:   return 2.0 + 0.8 * (crowd_size - 5_000) / 15_000
    if crowd_size < 100_000:  return 2.8 + 0.9 * (crowd_size - 20_000) / 80_000
    return min(4.2, 3.7 + 0.5 * (crowd_size - 100_000) / 100_000)

# ─── Event-type coefficients (from Astram public_event cause patterns) ────────
EVENT_TYPE_PARAMS: Dict[str, Dict] = {
    "cricket_match": dict(
        base_officers=12, base_barricades=8, surge_factor=1.0,
        pre_event_hrs=2.0, post_event_hrs=1.5,
        typical_radius_km=3.0,
        diversion_routes=["Via Richmond Road", "Via Hosur Road flyover"],
        desc="High-attendance, predictable start/end times. Heavy egress surge.",
    ),
    "concert_music": dict(
        base_officers=10, base_barricades=6, surge_factor=0.95,
        pre_event_hrs=1.5, post_event_hrs=2.0,
        typical_radius_km=2.5,
        diversion_routes=["Via MG Road", "Via residency road"],
        desc="Late-night events cause post-event peak. Night Hawkes excitation elevated.",
    ),
    "festival_religious": dict(
        base_officers=18, base_barricades=14, surge_factor=1.15,
        pre_event_hrs=3.0, post_event_hrs=3.0,
        typical_radius_km=5.0,
        diversion_routes=["Via Outer Ring Road", "Via alternate arterial"],
        desc="Multi-day footprint. Unpredictable crowd flow. High spillover risk.",
    ),
    "political_rally": dict(
        base_officers=22, base_barricades=16, surge_factor=1.2,
        pre_event_hrs=3.0, post_event_hrs=2.0,
        typical_radius_km=4.0,
        diversion_routes=["Via ORR North 1", "Via Tumkur Road bypass"],
        desc="High VIP security load. Procession routes amplify corridor contagion.",
    ),
    "sports_marathon": dict(
        base_officers=15, base_barricades=20, surge_factor=0.9,
        pre_event_hrs=2.0, post_event_hrs=1.0,
        typical_radius_km=8.0,
        diversion_routes=["Route closures — full corridor blocks required"],
        desc="Corridor closures rather than congestion. Linear impact zone.",
    ),
    "exhibition_trade": dict(
        base_officers=8, base_barricades=5, surge_factor=0.85,
        pre_event_hrs=1.0, post_event_hrs=1.5,
        typical_radius_km=2.0,
        diversion_routes=["Via parallel service road"],
        desc="Spread across business hours. Moderate sustained surge.",
    ),
    "vip_movement": dict(
        base_officers=25, base_barricades=10, surge_factor=2.8,
        pre_event_hrs=0.5, post_event_hrs=0.5,
        typical_radius_km=6.0,
        diversion_routes=["All parallel routes — coordinate with protocol"],
        desc="Short but intense. Entire corridors blocked. Transfer entropy maximised.",
    ),
    "procession_religious": dict(
        base_officers=14, base_barricades=12, surge_factor=1.1,
        pre_event_hrs=1.5, post_event_hrs=1.5,
        typical_radius_km=4.0,
        diversion_routes=["Via cross streets", "Via parallel road"],
        desc="Moving event → impact propagates along route corridor in sequence.",
    ),
    "construction": dict(
        base_officers=8, base_barricades=22, surge_factor=0.85,
        pre_event_hrs=0.0, post_event_hrs=0.0,
        typical_radius_km=2.0,
        diversion_routes=["Use nearest parallel/arterial corridors", "VMS signs route advisory"],
        desc="Long-term planned infrastructure work (dominant planned event type). Causes significant lane reduction, bottleneck behavior, and elevated background rates.",
    ),
}

# ─── Corridor baseline rates (events/hour from Astram, business hours) ────────
# Internal baseline rates for simulation (hardcoded approximations).
# App.py visualization uses Hawkes-derived rates (more accurate).
CORRIDOR_BASELINE_RATE: Dict[str, float] = {
    "Mysore Road": 0.52, "Bellary Road 1": 0.41, "Tumkur Road": 0.35,
    "Bellary Road 2": 0.25, "Hosur Road": 0.16, "ORR North 1": 0.14,
    "Old Madras Road": 0.17, "Magadi Road": 0.16, "ORR East 1": 0.14,
    "ORR North 2": 0.12, "West of Chord Road": 0.13, "ORR West 1": 0.11,
    "Bannerghatta Road": 0.12,
}

# ─── Hawkes branching ratios (from M3 fitted results) ─────────────────────────
HAWKES_N: Dict[str, float] = {
    "Mysore Road": 0.219, "Bellary Road 1": 0.121,
    "Tumkur Road": 0.05,  "Bellary Road 2": 0.05,
}


@dataclass
class EventSimResult:
    event_type:          str
    venue:               str
    crowd_size:          int
    start_hour:          float
    end_hour:            float
    affected_corridors:  List[str]

    # Timing
    pre_event_window:    str = ""
    peak_window:         str = ""
    post_event_window:   str = ""

    # Surge estimates
    surge_multiplier:    float = 1.0
    expected_extra_events_per_corridor: float = 0.0
    peak_congestion_intensity: str = "LOW"   # LOW / MEDIUM / HIGH / CRITICAL

    # Resource recommendations
    officers_required:   int = 0
    barricades_required: int = 0
    diversion_routes:    List[str] = field(default_factory=list)
    road_closure_zones:  List[str] = field(default_factory=list)

    # Hawkes warning
    hawkes_alert:        Optional[str] = None
    cascade_risk:        str = "LOW"

    # Narrative
    policy_notes:        List[str] = field(default_factory=list)

    # VMS broadcast text (pre-formatted for Variable Message Signs)
    vms_message:         str = ""


def generate_vms_text(
    event_type: str,
    intensity: str,
    corridor: str,
    diversion_route: str,
    peak_window: str,
    n_hawkes: float = 0.0
) -> str:
    vms_code_map = {'CRITICAL': 'CODE RED', 'HIGH': 'CODE AMBER', 'MEDIUM': 'CODE YELLOW', 'LOW': 'ALL CLEAR'}
    code_str = vms_code_map.get(intensity, 'CODE AMBER')
    cascade_warn = f"\n⚡ CASCADE n={n_hawkes:.2f}" if n_hawkes > 0.2 else ""
    return (
        f"*** {code_str} ***\n"
        f"MAJOR EVENT: {event_type.replace('_',' ').upper()}\n"
        f"Expect heavy traffic on:\n{corridor}\n"
        f"DIVERT VIA: {diversion_route[:45]}\n"
        f"Peak: {peak_window}{cascade_warn}"
    )


def simulate_event(
    event_type:  str,
    venue:       str,
    crowd_size:  int,
    start_hour:  float,
    end_hour:    float,
    day_of_week: str = "Sat",
) -> EventSimResult:
    """
    Core simulation function.

    Parameters
    ----------
    event_type  : one of EVENT_TYPE_PARAMS keys
    venue       : one of VENUE_CORRIDORS keys
    crowd_size  : expected attendance
    start_hour  : event start (24h float, e.g. 14.5 = 2:30 PM)
    end_hour    : event end
    day_of_week : Mon–Sun (weekends increase baseline by 1.3×)
    """
    params = EVENT_TYPE_PARAMS.get(event_type, EVENT_TYPE_PARAMS["festival_religious"])
    corridors = VENUE_CORRIDORS.get(venue, ["Mysore Road"])

    # ── Surge multiplier ──────────────────────────────────────────────────────
    crowd_surge  = crowd_to_surge(crowd_size)
    event_surge  = params["surge_factor"]
    weekend_mult = 1.3 if day_of_week in ("Sat", "Sun") else 1.0
    night_mult   = 1.4 if (start_hour >= 20 or end_hour <= 6) else 1.0
    total_surge  = crowd_surge * event_surge * weekend_mult * night_mult

    # ── Expected extra events ─────────────────────────────────────────────────
    duration_hrs = max(0.5, end_hour - start_hour)
    extra_events = []
    for corr in corridors:
        base = CORRIDOR_BASELINE_RATE.get(corr, 0.15)
        extra = base * (total_surge - 1.0) * duration_hrs
        extra_events.append(extra)
    avg_extra = float(np.mean(extra_events))

    # ── Peak intensity label ──────────────────────────────────────────────────
    if total_surge > 3.5:
        intensity = "CRITICAL"
    elif total_surge > 2.5:
        intensity = "HIGH"
    elif total_surge > 1.7:
        intensity = "MEDIUM"
    else:
        intensity = "LOW"

    # ── Resource scaling ──────────────────────────────────────────────────────
    scale = max(1.0, crowd_size / 10_000)
    officers   = int(params["base_officers"]   * min(scale, 4))
    barricades = int(params["base_barricades"] * min(scale, 3.5))

    # Pre/peak/post windows
    pre_start  = max(0, start_hour - params["pre_event_hrs"])
    post_end   = min(24, end_hour + params["post_event_hrs"])

    # ── Hawkes cascade risk ───────────────────────────────────────────────────
    max_n = max((HAWKES_N.get(c, 0.05) for c in corridors), default=0.05)
    # Planned event pre-excites the system: base λ multiplied by surge
    effective_excitation = max_n * total_surge
    if effective_excitation > 0.4:
        cascade_risk = "CRITICAL"
        hawkes_alert = (
            f"Branching ratio under event conditions ≈ {effective_excitation:.2f} → "
            "NEAR-UNSTABLE. Pre-position resources 2h before start."
        )
    elif effective_excitation > 0.25:
        cascade_risk = "HIGH"
        hawkes_alert = (
            f"Effective excitation ≈ {effective_excitation:.2f}. "
            "Event will trigger secondary breakdown cascade. Deploy 30min early."
        )
    elif effective_excitation > 0.15:
        cascade_risk = "MEDIUM"
        hawkes_alert = f"Moderate cascade risk (n·surge ≈ {effective_excitation:.2f})."
    else:
        cascade_risk = "LOW"
        hawkes_alert = None

    # ── Road closure zones ────────────────────────────────────────────────────
    closure_zones = []
    if crowd_size >= 30_000 or event_type in ("sports_marathon", "political_rally", "vip_movement"):
        closure_zones = [f"{c} — entry restriction within {params['typical_radius_km']:.0f} km"
                         for c in corridors]

    # ── Policy notes ──────────────────────────────────────────────────────────
    notes = [params["desc"]]
    if total_surge > 2.5:
        notes.append(
            f"ATLAS pre-excites Hawkes model for {corridors}: "
            f"background rate μ multiplied by {total_surge:.1f}× starting "
            f"{params['pre_event_hrs']:.0f}h before event."
        )
    if effective_excitation > 0.25:
        notes.append(
            "TE contagion network activated: downstream corridors should "
            "expect spill-over 15–30 min after primary surge."
        )
    notes.append(
        f"ResourceRAG pre-loaded with {event_type} diversion templates. "
        "Dispatch Assistant will surface these automatically."
    )

    # Build VMS message
    primary_corr = corridors[0] if corridors else 'affected corridor'
    primary_div  = params["diversion_routes"][0] if params["diversion_routes"] else 'ALT ROUTE'
    primary_n    = HAWKES_N.get(primary_corr, 0.0)
    vms_msg = generate_vms_text(
        event_type      = event_type,
        intensity       = intensity,
        corridor        = primary_corr,
        diversion_route = primary_div,
        peak_window     = f"{_hhmm(start_hour)} – {_hhmm(end_hour)}",
        n_hawkes        = primary_n
    )

    return EventSimResult(
        event_type          = event_type,
        venue               = venue,
        crowd_size          = crowd_size,
        start_hour          = start_hour,
        end_hour            = end_hour,
        affected_corridors  = corridors,
        pre_event_window    = f"{_hhmm(pre_start)} – {_hhmm(start_hour)}",
        peak_window         = f"{_hhmm(start_hour)} – {_hhmm(end_hour)}",
        post_event_window   = f"{_hhmm(end_hour)} – {_hhmm(post_end)}",
        surge_multiplier            = round(total_surge, 2),
        expected_extra_events_per_corridor = round(avg_extra, 1),
        peak_congestion_intensity   = intensity,
        officers_required           = officers,
        barricades_required         = barricades,
        diversion_routes            = params["diversion_routes"],
        road_closure_zones          = closure_zones,
        hawkes_alert                = hawkes_alert,
        cascade_risk                = cascade_risk,
        policy_notes                = notes,
        vms_message                 = vms_msg,
    )


def simulate_example_events(df: pd.DataFrame) -> List[dict]:
    """Run canonical planned-event scenarios and save results."""
    scenarios = [
        dict(event_type="construction",      venue="Palace Grounds",
             crowd_size=0,     start_hour=8,   end_hour=18, day_of_week="Mon"),
        dict(event_type="cricket_match",     venue="Chinnaswamy Stadium",
             crowd_size=50_000, start_hour=14, end_hour=20, day_of_week="Sun"),
        dict(event_type="political_rally",   venue="Freedom Park",
             crowd_size=25_000, start_hour=16, end_hour=20, day_of_week="Sun"),
        dict(event_type="festival_religious",venue="Lalbagh Botanical Garden",
             crowd_size=75_000, start_hour=8,  end_hour=22, day_of_week="Sun"),
        dict(event_type="concert_music",     venue="Palace Grounds",
             crowd_size=15_000, start_hour=18, end_hour=23, day_of_week="Sat"),
        dict(event_type="vip_movement",      venue="Vidhana Soudha",
             crowd_size=500,   start_hour=10,  end_hour=12, day_of_week="Mon"),
    ]

    results = []
    print("[M8] Event Impact Simulator — calibrated from Astram planned events")
    print()
    for s in scenarios:
        r = simulate_event(**s)
        results.append(asdict(r))
        print(f"  [{r.event_type.upper()}] @ {r.venue}")
        print(f"    Crowd: {r.crowd_size:,}  |  Surge: {r.surge_multiplier:.1f}×  |  "
              f"Intensity: {r.peak_congestion_intensity}")
        print(f"    Officers: {r.officers_required}  |  Barricades: {r.barricades_required}  |  "
              f"Cascade risk: {r.cascade_risk}")
        print(f"    Peak window: {r.peak_window}")
        print()

    out = MODEL_DIR / "event_sim_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[M8] ✅ Saved {len(results)} scenario results → {out}")
    return results


def run_event_simulator(df: pd.DataFrame) -> List[dict]:
    return simulate_example_events(df)


if __name__ == "__main__":
    df = pd.read_parquet(DATA_DIR / "astram_clean.parquet")
    run_event_simulator(df)
