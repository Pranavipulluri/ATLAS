"""
bot/nlp_parser.py
Natural-language incident parser for ATLAS Dispatch Bot.

Pure keyword-based — no LLM dependency, works offline in police control rooms.
Parses free-text voice transcripts or typed messages into structured event dicts
ready for M6 ResourceRAG or M8 Event Simulator.
"""

import re
from datetime import datetime
from typing import Optional, Tuple

# ─── Corridor keyword map ─────────────────────────────────────────────────────
CORRIDOR_KEYWORDS = {
    "Mysore Road":          ["mysore road", "mysore", "kengeri", "rajarajeshwari", "bidadi"],
    "Bellary Road 1":       ["bellary road", "bellary", "hebbal", "yelahanka", "airport road", "thanisandra"],
    "Bellary Road 2":       ["bellary 2", "devanahalli", "international airport"],
    "Tumkur Road":          ["tumkur road", "tumkur", "peenya", "nelamangala", "nagasandra"],
    "Hosur Road":           ["hosur road", "hosur", "electronic city", "bommanahalli",
                             "koramangala", "silk board", "hsr layout"],
    "ORR North 1":          ["orr north", "outer ring road north", "hebbal flyover", "esteem"],
    "ORR East 1":           ["orr east", "outer ring road east", "marathahalli", "bellandur"],
    "ORR North 2":          ["orr north 2", "kogilu", "jakkur"],
    "ORR West 1":           ["orr west", "rajajinagar", "chord road junction"],
    "Old Madras Road":      ["old madras", "indiranagar", "tin factory", "kr puram"],
    "Magadi Road":          ["magadi road", "magadi", "vijayanagar", "chord road"],
    "West of Chord Road":   ["west chord", "west of chord"],
    "Bannerghatta Road":    ["bannerghatta", "jp nagar", "hulimavu"],
}

# ─── Cause keyword map ────────────────────────────────────────────────────────
# Cause priority: incident causes beat event causes when both match
CAUSE_PRIORITY = {
    "accident":          10,
    "vehicle_breakdown": 9,
    "tree_fall":          8,
    "water_logging":      7,
    "pot_holes":          6,
    "construction":       5,
    "vip_movement":       4,
    "procession":         3,
    "public_event":       2,
}

CAUSE_KEYWORDS = {
    "vehicle_breakdown": [
        "breakdown", "broke down", "broken down", "stalled", "stall",
        "tow", "engine failure", "flat tyre", "puncture", "dead battery",
        "overheated", "smoke", "vehicle stuck", "bus stuck", "truck stuck",
    ],
    "accident": [
        "accident", "crash", "collision", "hit", "rammed", "overturned",
        "rollover", "injured", "casualties", "multiple vehicles", "chain accident",
    ],
    "pot_holes": [
        "pothole", "pot hole", "crater", "road damage", "damaged road",
        "road cavity", "sinking road",
    ],
    "water_logging": [
        "waterlog", "water logging", "flooded", "flood", "heavy rain",
        "rainfall", "inundated", "drainage overflow", "rainwater",
    ],
    "construction": [
        "construction", "road work", "digging", "pipe laying", "cable work",
        "metro work", "utility work", "excavation",
    ],
    "tree_fall": [
        "tree fall", "fallen tree", "tree on road", "uprooted tree",
        "tree branch", "tree collapsed",
    ],
    "public_event": [
        "cricket", "match", "ipl", "concert", "show", "exhibition",
        "event", "gathering", "sports",
    ],
    "vip_movement": [
        "vip", "convoy", "president", "prime minister", "pm convoy",
        "cm convoy", "chief minister", "dignitary", "protocol",
    ],
    "procession": [
        "procession", "rally", "protest", "demonstration", "march",
        "bandh", "hartal", "agitation",
    ],
}

# ─── Vehicle keyword map ──────────────────────────────────────────────────────
VEH_KEYWORDS = {
    "bmtc_bus":      ["bmtc", "city bus", "public bus", "metropolitan bus"],
    "ksrtc_bus":     ["ksrtc", "state bus", "interstate bus"],
    "private_bus":   ["private bus", "school bus", "tourist bus", "volvo"],
    "heavy_vehicle": ["heavy vehicle", "lorry", "tanker", "tipper", "container", "trailer"],
    "truck":         ["truck", "mini truck", "tempo"],
    "lcv":           ["lcv", "auto", "three wheeler", "tempo traveller", "minivan"],
    "private_car":   ["car", "suv", "sedan", "vehicle", "private vehicle"],
    "two_wheeler":   ["bike", "motorcycle", "scooter", "two wheeler"],
}

# ─── Priority triggers ────────────────────────────────────────────────────────
HIGH_PRIORITY_TRIGGERS = [
    "accident", "crash", "injured", "casualt", "heavy vehicle", "truck",
    "highway", "multiple", "chain", "vip", "convoy", "flood", "critical",
    "blocked completely", "total blockage",
]

# ─── Planned event triggers ───────────────────────────────────────────────────
PLANNED_EVENT_TRIGGERS = [
    "tomorrow", "next week", "this weekend", "tonight at", "scheduled",
    "cricket match", "ipl", "concert", "political rally", "procession",
    "festival", "expected", "crowd of", "expected crowd", "thousand people",
]

# ─── Event type for M8 ────────────────────────────────────────────────────────
M8_EVENT_TYPE_KEYWORDS = {
    "cricket_match":       ["cricket", "ipl", "test match", "t20", "one day"],
    "concert_music":       ["concert", "music", "dj", "band", "singer"],
    "festival_religious":  ["festival", "puja", "ganesh", "diwali", "eid", "christmas", "navratri"],
    "political_rally":     ["rally", "political", "election", "campaign", "party meeting"],
    "sports_marathon":     ["marathon", "run", "race", "cycling"],
    "vip_movement":        ["vip", "convoy", "pm", "cm", "president"],
    "procession_religious":["procession", "rath yatra", "parade", "march"],
    "exhibition_trade":    ["exhibition", "trade fair", "expo", "auto show"],
    "construction":        ["construction", "road work", "metro work", "utility work", "digging"],
}

# ─── Crowd size extraction ────────────────────────────────────────────────────
CROWD_PATTERNS = [
    r'(\d[\d,]*)\s*(?:thousand|k)\s*(?:people|crowd|attendees|visitors)?',
    r'(\d[\d,]*)\s*(?:lakh)\s*(?:people|crowd)?',
    r'crowd\s+of\s+(\d[\d,]*)',
    r'expected\s+(\d[\d,]+)',
    r'expecting\s+(\d[\d,]+)',
    r'about\s+(\d{4,})',
    r'around\s+(\d{4,})',
    r'(\d{4,})\s*(?:people|attendees|visitors|crowd)',
    r'(\d{5,})',   # bare 5+ digit number is almost certainly a crowd size
]

# ─── Time extraction ──────────────────────────────────────────────────────────
TIME_PATTERNS = [
    r'(\d{1,2})[:\.](\d{2})\s*(am|pm)',
    r'(\d{1,2})\s*(am|pm)',
    r'at\s+(\d{1,2})\s*(am|pm)?',
]


def _extract_corridor(text: str) -> Tuple[str, float]:
    """Return (corridor_name, confidence 0–1)."""
    text_lower = text.lower()
    best_corr, best_score = "Mysore Road", 0.0
    for corridor, keywords in CORRIDOR_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                score = len(kw) / 20.0  # longer match = more specific
                if score > best_score:
                    best_score, best_corr = score, corridor
    return best_corr, min(1.0, best_score * 3)


def _extract_cause(text: str) -> Tuple[str, float]:
    """Return (cause_clean, confidence 0–1). Higher-priority incident causes win."""
    text_lower = text.lower()
    scores = {}  # cause -> keyword_length_sum
    for cause, keywords in CAUSE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[cause] = scores.get(cause, 0) + len(kw)
    if not scores:
        return "vehicle_breakdown", 0.3
    # Weight by priority so incident causes beat event causes
    weighted = {c: scores[c] * CAUSE_PRIORITY.get(c, 1) for c in scores}
    best = max(weighted, key=weighted.get)
    return best, min(1.0, scores[best] / 15.0)


def _extract_vehicle(text: str) -> str:
    text_lower = text.lower()
    for vtype, keywords in VEH_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return vtype
    return "unknown"


def _extract_priority(text: str, cause: str) -> str:
    text_lower = text.lower()
    for trigger in HIGH_PRIORITY_TRIGGERS:
        if trigger in text_lower:
            return "High"
    if cause in ("accident", "vip_movement", "water_logging"):
        return "High"
    return "Low"


def _extract_hour(text: str) -> int:
    """Try to extract hour from text, fall back to current hour."""
    text_lower = text.lower()
    for pattern in TIME_PATTERNS:
        m = re.search(pattern, text_lower)
        if m:
            try:
                hr = int(m.group(1))
                groups = m.groups()
                ampm = next((g for g in groups[1:] if g and g in ('am', 'pm')), None)
                if ampm == 'pm' and hr < 12:
                    hr += 12
                elif ampm == 'am' and hr == 12:
                    hr = 0
                return hr % 24
            except Exception:
                pass
    return datetime.now().hour


def _extract_crowd_size(text: str) -> Optional[int]:
    text_lower = text.lower().replace(',', '')
    for pattern in CROWD_PATTERNS:
        m = re.search(pattern, text_lower)
        if m:
            try:
                n = int(m.group(1).replace(',', ''))
                if 'lakh' in text_lower[max(0, m.start()-5):m.end()+5]:
                    n *= 100_000
                elif any(k in text_lower[max(0, m.start()-2):m.end()+5]
                         for k in ['thousand', 'k']):
                    n *= 1_000
                return n
            except Exception:
                pass
    return None


def _is_planned_event(text: str) -> bool:
    """Only treat as planned if strong planned-event temporal signal AND no strong incident keywords."""
    text_lower = text.lower()
    has_planned = any(t in text_lower for t in PLANNED_EVENT_TRIGGERS)
    # If strong incident language present, don't redirect to M8
    incident_triggers = ["breakdown", "accident", "crash", "tree", "flood", "pothole",
                         "waterlog", "construction", "tow", "blocked", "stuck"]
    has_incident = any(t in text_lower for t in incident_triggers)
    return has_planned and not has_incident


def _extract_m8_event_type(text: str) -> str:
    text_lower = text.lower()
    for etype, keywords in M8_EVENT_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return etype
    return "festival_religious"


def _extract_m8_venue(text: str) -> str:
    from src.m8_event_simulator import VENUE_CORRIDORS
    text_lower = text.lower()
    venue_keywords = {
        "Chinnaswamy Stadium":       ["chinnaswamy", "ksca", "m.chinnaswamy"],
        "Kanteerava Stadium":        ["kanteerava"],
        "Lalbagh Botanical Garden":  ["lalbagh"],
        "Cubbon Park":               ["cubbon"],
        "Palace Grounds":            ["palace grounds", "palace ground"],
        "Freedom Park":              ["freedom park"],
        "Vidhana Soudha":            ["vidhana soudha", "vidhana", "secretariat"],
        "Forum Mall (Koramangala)":  ["forum mall", "koramangala mall"],
        "Orion Mall (Rajajinagar)":  ["orion mall"],
    }
    for venue, kws in venue_keywords.items():
        for kw in kws:
            if kw in text_lower:
                return venue
    return "Custom Venue"


def parse_incident(text: str) -> dict:
    """
    Parse free-text incident report into structured dict.

    Returns a dict with keys:
      - mode: 'dispatch' | 'simulate'
      - event_dict: dict ready for M6 query() or M8 simulate_event()
      - confidence: float 0–1
      - extracted: dict of what was found
    """
    corridor, corr_conf = _extract_corridor(text)
    cause,    cause_conf = _extract_cause(text)
    veh_type  = _extract_vehicle(text)
    priority  = _extract_priority(text, cause)
    hour      = _extract_hour(text)
    is_night  = hour >= 20 or hour < 6
    is_planned = _is_planned_event(text)
    crowd_size = _extract_crowd_size(text)

    # ── Route to M8 if planned event ─────────────────────────────────────────
    if is_planned or cause in ("public_event", "vip_movement", "procession"):
        event_type = _extract_m8_event_type(text)
        venue      = _extract_m8_venue(text)
        return {
            "mode": "simulate",
            "event_dict": {
                "event_type":  event_type,
                "venue":       venue,
                "crowd_size":  crowd_size or 10_000,
                "start_hour":  float(hour),
                "end_hour":    float((hour + 4) % 24),
                "day_of_week": _get_day(text),
            },
            "confidence": cause_conf * 0.8,
            "extracted": {
                "type": event_type, "venue": venue,
                "crowd": crowd_size, "hour": hour,
            },
        }

    # ── Route to M6 for live incidents ───────────────────────────────────────
    event_class = "acute" if cause in (
        "vehicle_breakdown", "accident", "tree_fall"
    ) else "chronic" if cause in (
        "construction", "pot_holes", "water_logging"
    ) else "planned"

    return {
        "mode": "dispatch",
        "event_dict": {
            "event_cause_clean":   cause,
            "corridor":            corridor,
            "priority":            priority,
            "requires_road_closure": "TRUE" if priority == "High" and cause == "accident" else "FALSE",
            "veh_type_imputed":    veh_type,
            "event_class":         event_class,
            "hour_IST":            hour,
            "day_of_week":         datetime.now().weekday(),
            "month":               datetime.now().month,
            "is_night":            is_night,
            "event_type":          "unplanned",
        },
        "confidence": min(0.95, (corr_conf + cause_conf) / 2),
        "extracted": {
            "corridor": corridor, "cause": cause,
            "vehicle": veh_type, "priority": priority,
            "hour": hour, "is_night": is_night,
        },
    }


def _get_day(text: str) -> str:
    text_lower = text.lower()
    for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
        if day in text_lower:
            return day[:3].capitalize()
    if "weekend" in text_lower or "saturday" in text_lower:
        return "Sat"
    if "tomorrow" in text_lower:
        tomorrow = (datetime.now().weekday() + 1) % 7
        return ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][tomorrow]
    return ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][datetime.now().weekday()]
