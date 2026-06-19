"""
Deterministic Priority Auction Engine
P = (w1 * E) + (w2 * D) + (w3 * T_wait)

This is the FAST math engine that makes all real-time signal decisions.
No LLMs involved - pure deterministic computation.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


# Auction Weights
W_EMERGENCY = 10.0   # w1: emergency multiplier
W_DENSITY = 2.0      # w2: vehicle density weight
W_WAIT = 0.5         # w3: wait time weight

# Emergency flag value
EMERGENCY_VALUE = 1000  # E = 1000 when emergency, 0 otherwise

# Phase timing constants
MIN_GREEN_TIME = 5.0    # seconds minimum green before switching
MAX_GREEN_TIME = 45.0   # seconds max green (starvation prevention)
GREEN_WAVE_BOOST = 500  # extra priority for lanes expecting emergency


@dataclass
class LaneScore:
    """Result of the priority auction for a single lane."""
    lane: str
    score: float
    density: int
    wait_time: float
    has_emergency: bool
    green_wave_boost: float = 0.0
    has_pedestrians: bool = False


def compute_lane_priority(
    lane: str,
    density: int,
    wait_time: float,
    has_emergency: bool,
    green_wave_boost: float = 0.0,
    has_pedestrians: bool = False
) -> LaneScore:
    """
    Compute the priority score for a single lane.
    P = (w1 * E) + (w2 * D) + (w3 * T_wait) + green_wave_boost + pedestrian_boost
    """
    e = EMERGENCY_VALUE if has_emergency else 0
    p_boost = (50.0 * wait_time) if has_pedestrians else 0.0
    score = (W_EMERGENCY * e) + (W_DENSITY * density) + (W_WAIT * wait_time) + green_wave_boost + p_boost
    return LaneScore(
        lane=lane,
        score=score,
        density=density,
        wait_time=wait_time,
        has_emergency=has_emergency,
        green_wave_boost=green_wave_boost,
        has_pedestrians=has_pedestrians
    )


def run_auction(lane_scores: List[LaneScore]) -> List[LaneScore]:
    """
    Run the priority auction. Returns lanes sorted by priority (highest first).
    """
    return sorted(lane_scores, key=lambda x: x.score, reverse=True)


def should_switch_phase(
    current_green: str,
    ranked_lanes: List[LaneScore],
    time_in_phase: float
) -> Tuple[bool, str, str]:
    """
    Determine if the signal should switch phase.
    
    Returns: (should_switch, target_lane, reason)
    """
    winner = ranked_lanes[0]
    
    # Rule 1: MAX GREEN OVERRIDE (starvation prevention)
    if time_in_phase >= MAX_GREEN_TIME:
        # Find the lane with longest wait time that isn't current green
        non_green = [l for l in ranked_lanes if l.lane != current_green]
        if non_green:
            starved = max(non_green, key=lambda x: x.wait_time)
            return (True, starved.lane, 
                    f"MAX_GREEN_OVERRIDE: {current_green} exceeded {MAX_GREEN_TIME}s. "
                    f"Rotating to {starved.lane} (waited {starved.wait_time:.1f}s)")
    
    # Rule 2: EMERGENCY PRE-EMPT (ignore min green)
    if winner.has_emergency and winner.lane != current_green:
        return (True, winner.lane,
                f"EMERGENCY_PREEMPT: Ambulance detected on {winner.lane} lane. "
                f"Priority score: {winner.score:.1f}. Immediate phase switch.")
    
    # Rule 3: GREEN WAVE PRE-EMPT (from A2A alert, ignore min green)
    if winner.green_wave_boost > 0 and winner.lane != current_green:
        return (True, winner.lane,
                f"GREEN_WAVE_PREEMPT: Incoming emergency vehicle alert from neighbor. "
                f"Pre-clearing {winner.lane} lane. Boost: +{winner.green_wave_boost:.0f}")
    
    # Rule 4: MIN GREEN guard
    if time_in_phase < MIN_GREEN_TIME:
        return (False, current_green,
                f"MAINTAIN: Minimum green time not reached ({time_in_phase:.1f}s < {MIN_GREEN_TIME}s)")
    
    # Rule 5: STANDARD AUCTION
    if winner.lane != current_green:
        return (True, winner.lane,
                f"AUCTION_SWITCH: {winner.lane} wins auction with score {winner.score:.1f} "
                f"(D={winner.density}, T={winner.wait_time:.1f}s)")
    
    return (False, current_green,
            f"MAINTAIN: {current_green} still has highest priority ({winner.score:.1f})")
