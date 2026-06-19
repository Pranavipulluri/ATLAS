"""
src/radio_dispatch.py — ATLAS GenAI Radio Dispatch Translator

Converts M6 ResourceRAG JSON dispatch cards into walkie-talkie-ready
radio scripts using Gemini 1.5 Flash.  Outputs English + Kannada.

Usage:
    from src.radio_dispatch import generate_radio_script
    scripts = generate_radio_script(card, extracted_event)

Environment:
    GEMINI_API_KEY  — Google AI Studio key (or uses GOOGLE_API_KEY)
"""

import os
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── Attempt Gemini import ──────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

_model = None


def _get_api_key() -> str:
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENERATIVEAI_API_KEY"):
        val = os.environ.get(key, "")
        if val:
            return val
    # Try .env file
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            for prefix in ("GEMINI_API_KEY=", "GOOGLE_API_KEY="):
                if line.startswith(prefix):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _init_model():
    global _model
    if _model is not None:
        return True
    if not _GEMINI_AVAILABLE:
        return False
    api_key = _get_api_key()
    if not api_key:
        return False
    try:
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel("gemini-1.5-flash")
        return True
    except Exception:
        return False


# ── Prompt template ────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are ATLAS Radio Dispatch AI, the last-mile translation layer for a traffic control system in Bengaluru, India.

Your job: Convert a structured JSON dispatch card into the EXACT radio script a stressed control-room operator will read aloud into a walkie-talkie.

Rules for the radio script:
1. START with "All units, Code [HIGH/MEDIUM/LOW]."
2. State: location, cause, vehicle type (if known).
3. State dispatch action: "Dispatching X tow trucks / patrol units."
4. State clearance estimate: "Expecting X-minute clearance."
5. If cascade risk is HIGH or CRITICAL: alert nearby corridor units to position at junctions.
6. End with: "Acknowledge on Channel 3. ATLAS out."
7. Keep it under 60 words. No filler. Operators are under stress.
8. Be authoritative, clear, actionable.

Output ONLY valid JSON with exactly these two keys:
{
  "english": "<radio script in English>",
  "kannada": "<same script translated into Kannada, using Kannada script, ಸರ್ಕಾರಿ ಭಾಷೆ style>"
}

Do NOT include markdown, do NOT include any text outside the JSON object.
"""


def _build_card_summary(card: dict, extracted: dict) -> str:
    """Flatten the M6 card + extracted event into a plain-text summary for the LLM."""
    cause    = extracted.get("cause", "unknown").replace("_", " ")
    corridor = extracted.get("corridor", "unknown")
    vehicle  = extracted.get("vehicle", "unknown").replace("_", " ")
    priority = card.get("recommended_priority", "High")
    res_mins = int(card.get("expected_resolution_mins") or card.get("expected_resolution_hrs", 1) * 60 or 60)
    closure  = card.get("road_closure_recommended", False)
    cascade  = card.get("cascade_risk", "LOW")
    conf     = card.get("confidence_label", "MEDIUM")
    iqr      = card.get("resolution_iqr_mins", [res_mins * 0.7, res_mins * 1.3])
    quality  = card.get("intervention_quality", "GOOD")

    lines = [
        f"PRIORITY: {priority.upper()}",
        f"CAUSE: {cause}",
        f"VEHICLE INVOLVED: {vehicle}",
        f"LOCATION: {corridor}",
        f"EXPECTED CLEARANCE: {res_mins:.0f} minutes (IQR: {iqr[0]:.0f}–{iqr[1]:.0f} min)",
        f"ROAD CLOSURE REQUIRED: {'Yes' if closure else 'No'}",
        f"CASCADE RISK: {cascade}",
        f"HISTORICAL INTERVENTION QUALITY: {quality}",
        f"RAG CONFIDENCE: {conf}",
    ]

    # Hawkes cascade alert
    hawkes_alert = extracted.get("hawkes_alert", "")
    if hawkes_alert:
        lines.append(f"HAWKES ALERT: {hawkes_alert}")

    # TE downstream risk
    te_downstream = extracted.get("te_downstream", [])
    if te_downstream:
        lines.append(f"CONTAGION RISK TO: {', '.join(te_downstream[:2])}")

    return "\n".join(lines)


def generate_radio_script(card: dict, extracted: dict) -> dict:
    """
    Generate walkie-talkie radio scripts from an M6 dispatch card.

    Parameters
    ----------
    card      : dict  — output of m6_resource_rag.query()
    extracted : dict  — parsed event dict (corridor, cause, vehicle, etc.)

    Returns
    -------
    dict with keys:
        'english'  : str  — radio script in English
        'kannada'  : str  — same script in Kannada
        'source'   : str  — 'gemini-1.5-flash' | 'fallback-template'
        'card_summary' : str — the plain-text card fed to the LLM
    """
    card_summary = _build_card_summary(card, extracted)

    # ── Try Gemini ─────────────────────────────────────────────────────────────
    if _init_model():
        try:
            prompt = f"{_SYSTEM_PROMPT}\n\nDISPATCH CARD:\n{card_summary}"
            response = _model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,        # low: consistent, not creative
                    "max_output_tokens": 512,
                    "candidate_count": 1,
                },
            )
            raw = response.text.strip()
            # Strip markdown fences if model added them
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
            parsed = json.loads(raw)
            return {
                "english":      parsed.get("english", ""),
                "kannada":      parsed.get("kannada", ""),
                "source":       "gemini-1.5-flash",
                "card_summary": card_summary,
            }
        except Exception:
            # Fall through to template fallback
            pass
    else:
        # Fall through to template fallback
        pass

    # ── Template fallback (works with zero dependencies) ──────────────────────
    return _template_fallback(card, extracted, card_summary)


def _template_fallback(card: dict, extracted: dict, card_summary: str) -> dict:
    """
    Deterministic template-based radio script when Gemini is unavailable.
    Covers enough variation for a convincing demo.
    """
    cause    = extracted.get("cause", "incident").replace("_", " ")
    corridor = extracted.get("corridor", "the affected corridor")
    vehicle  = extracted.get("vehicle", "unknown").replace("_", " ")
    priority = card.get("recommended_priority", "High").upper()
    res_mins = int(card.get("expected_resolution_mins") or card.get("expected_resolution_hrs", 1) * 60 or 60)
    closure  = card.get("road_closure_recommended", False)
    cascade  = card.get("cascade_risk", "LOW")
    te_down  = extracted.get("te_downstream", [])

    code = "HIGH" if priority == "HIGH" else "MEDIUM" if priority == "MEDIUM" else "LOW"

    # Tow truck logic
    tow_note = ""
    if any(v in cause for v in ["breakdown", "accident", "vehicle"]):
        tow_note = "Dispatching 1 tow truck and 2 patrol units. "
    else:
        tow_note = "Dispatching 2 patrol units and drainage crew. "

    closure_note = "Partial road closure in effect. " if closure else ""

    cascade_note = ""
    if cascade in ("HIGH", "CRITICAL") and te_down:
        cascade_note = f"{te_down[0]} units, position at junction to prevent spillover. "
    elif cascade in ("HIGH", "CRITICAL"):
        cascade_note = "Adjacent corridor units, position at junctions now. "

    english = (
        f"All units, Code {code}. "
        f"{cause.title()} on {corridor}. "
        f"{tow_note}"
        f"Expecting {res_mins}-minute clearance. "
        f"{closure_note}"
        f"{cascade_note}"
        f"Acknowledge on Channel 3. ATLAS out."
    ).strip()

    # Minimal Kannada template (key phrases)
    code_kn = {"HIGH": "ಹೈ", "MEDIUM": "ಮಧ್ಯಮ", "LOW": "ಕಡಿಮೆ"}.get(code, code)
    kannada = (
        f"ಎಲ್ಲಾ ಘಟಕಗಳಿಗೆ, ಕೋಡ್ {code_kn}. "
        f"{corridor} ನಲ್ಲಿ {cause.title()} ಘಟನೆ. "
        f"{res_mins} ನಿಮಿಷಗಳಲ್ಲಿ ತೆರವು ನಿರೀಕ್ಷಿಸಲಾಗಿದೆ. "
        f"ಚಾನೆಲ್ 3 ನಲ್ಲಿ ದೃಢೀಕರಿಸಿ. ATLAS ಔಟ್."
    )

    return {
        "english":      english,
        "kannada":      kannada,
        "source":       "fallback-template",
        "card_summary": card_summary,
    }


# ── CLI test harness ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    test_card = {
        "recommended_priority": "High",
        "road_closure_recommended": True,
        "expected_resolution": "50 minutes",
        "expected_resolution_mins": 50,
        "resolution_iqr_mins": [35, 75],
        "confidence_label": "HIGH",
        "intervention_quality": "EXCELLENT",
        "avg_intervention_success": 0.92,
        "cascade_risk": "HIGH",
        "n_similar_events": 8,
    }
    test_extracted = {
        "cause": "vehicle_breakdown",
        "corridor": "Mysore Road",
        "vehicle": "bmtc_bus",
        "hour": 18,
        "is_night": False,
        "hawkes_alert": "n=0.312 — high self-excitation. Secondary breakdown within 30 min likely.",
        "te_downstream": ["Bellary Road 1", "Tumkur Road"],
    }

    print("=" * 60)
    print("ATLAS Radio Dispatch — GenAI Last-Mile Translator")
    print("=" * 60)
    result = generate_radio_script(test_card, test_extracted)
    print(f"\nSource: {result['source']}")
    print(f"\n--- Card Summary fed to LLM ---\n{result['card_summary']}")
    print(f"\n📻 ENGLISH RADIO SCRIPT:\n{result['english']}")
    print(f"\n📻 ಕನ್ನಡ ರೇಡಿಯೋ ಸ್ಕ್ರಿಪ್ಟ್:\n{result['kannada']}")
