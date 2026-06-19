"""
bot/dispatch_bot.py — ATLAS Telegram Dispatch Bot

Usage:
    1. pip install pyTelegramBotAPI
    2. Set TELEGRAM_BOT_TOKEN in .env (or export as env var)
    3. python bot/dispatch_bot.py

Commands:
    (any text)      — Parse incident/event and get dispatch card
    /dispatch <msg> — Force M6 dispatch query
    /simulate <msg> — Force M8 event simulation
    /risk           — Current network risk snapshot
    /help           — Show usage

Pitch demo:
    Send audio message: "Heavy rainfall and a truck breakdown on Mysore Road"
    Bot replies with: 🚨 CODE HIGH dispatch card in < 2 seconds
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Lazy-load heavy models once at startup ────────────────────────────────────
_rag_loaded = False
_hawkes     = None
_te_data    = None

def _ensure_rag():
    global _rag_loaded
    if not _rag_loaded:
        from src.m6_resource_rag import load_index
        load_index()
        _rag_loaded = True

def _load_models():
    global _hawkes, _te_data
    hawkes_path = ROOT / "models" / "hawkes_results.json"
    te_path     = ROOT / "models" / "te_matrix.json"
    if hawkes_path.exists():
        with open(hawkes_path) as f:
            _hawkes = json.load(f)
    if te_path.exists():
        with open(te_path) as f:
            _te_data = json.load(f)

# ── Telegram setup ────────────────────────────────────────────────────────────
try:
    import telebot
    from telebot.types import Message
except ImportError:
    print("[ERROR] pyTelegramBotAPI not installed.")
    print("        Run: pip install pyTelegramBotAPI")
    sys.exit(1)

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
if not TOKEN:
    # Try .env file
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("TELEGRAM_BOT_TOKEN="):
                TOKEN = line.split("=", 1)[1].strip().strip('"').strip("'")
                break

if not TOKEN:
    print("[ERROR] TELEGRAM_BOT_TOKEN not set.")
    print("        Create a .env file with: TELEGRAM_BOT_TOKEN=your_token_here")
    print("        Get a token from @BotFather on Telegram.")
    sys.exit(1)

bot = telebot.TeleBot(TOKEN, parse_mode="HTML")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ATLAS-BOT] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Emoji + formatting helpers ────────────────────────────────────────────────
INTENSITY_EMOJI = {
    "CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢",
}
CASCADE_EMOJI = {
    "CRITICAL": "⚡", "HIGH": "🔥", "MEDIUM": "⚠️", "LOW": "✅",
}
QUALITY_EMOJI = {
    "EXCELLENT": "🏆", "GOOD": "✅", "POOR": "⚠️",
}

def _hr_time(h: float) -> str:
    hh = int(h) % 24
    mm = int((h % 1) * 60)
    suffix = "AM" if hh < 12 else "PM"
    hh12 = hh % 12 or 12
    return f"{hh12}:{mm:02d} {suffix}"


def format_dispatch_card(card: dict, extracted: dict) -> str:
    """Format M6 ResourceRAG result as a Telegram dispatch card."""
    conf_label = card.get("confidence_label", "LOW")
    conf_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(conf_label, "⚪")
    iq = card.get("intervention_quality", "GOOD")
    priority = card.get("recommended_priority", "High")
    pri_emoji = "🚨" if priority == "High" else "🔵"

    # Hawkes cascade check
    corr = extracted.get("corridor", "")
    cascade_line = ""
    if _hawkes and corr in _hawkes:
        n = _hawkes[corr].get("branching_ratio", 0)
        if n > 0.15:
            cascade_line = (
                f"\n⚡ <b>CASCADE RISK:</b> Branching ratio n={n:.3f} on {corr}\n"
                f"   → High probability of secondary breakdown within 30 min"
            )

    # TE downstream alert
    te_line = ""
    if _te_data and corr:
        sig = _te_data.get("significant_edges", [])
        downstreams = [e["dst"] for e in sig if e["src"] == corr][:2]
        if downstreams:
            te_line = f"\n🔀 <b>TE ALERT:</b> Monitor {', '.join(downstreams)} for contagion"

    res = card.get("expected_resolution", "Unknown")
    iqr = card.get("resolution_iqr_mins", [0, 0])
    closure = "✅ Required" if card.get("road_closure_recommended") else "❌ Not Required"

    msg = (
        f"🚦 <b>ATLAS DISPATCH CARD</b>\n"
        f"{'─'*32}\n"
        f"{pri_emoji} <b>PRIORITY: {priority.upper()}</b>  |  "
        f"{conf_emoji} Confidence: {conf_label}\n\n"
        f"📍 <b>Corridor:</b> {extracted.get('corridor', 'Unknown')}\n"
        f"⚠️  <b>Cause:</b> {extracted.get('cause', 'unknown').replace('_', ' ').title()}\n"
        f"🚗 <b>Vehicle:</b> {extracted.get('vehicle', 'unknown').replace('_', ' ').title()}\n"
        f"🕐 <b>Time:</b> {extracted.get('hour', 'now')}:00 IST "
        f"{'🌙 (Night)' if extracted.get('is_night') else '☀️'}\n\n"
        f"{'─'*32}\n"
        f"⏱ <b>Expected clearance:</b> {res}\n"
        f"📊 <b>Clearance window:</b> {iqr[0]:.0f}–{iqr[1]:.0f} min\n"
        f"🚧 <b>Road closure:</b> {closure}\n"
        f"{QUALITY_EMOJI[iq]} <b>Historical intervention quality:</b> {iq}\n"
        f"   (avg success score: {card.get('avg_intervention_success', 0):.2f})\n"
        f"{cascade_line}"
        f"{te_line}\n\n"
        f"{'─'*32}\n"
        f"📋 <b>Based on {card.get('n_similar_events', 0)} similar historical events</b>\n"
        f"{'⚠️ LOW SIMILARITY — use domain expertise' if card.get('low_confidence_warning') else ''}"
        f"{'♻️ Fallback mode (cause-only lookup)' if card.get('fallback_used') else ''}\n"
        f"<i>🤖 ATLAS M6 ResourceRAG · {datetime.now().strftime('%H:%M IST')}</i>"
    )
    return msg


def format_event_card(result) -> str:
    """Format M8 EventSimResult as a Telegram event impact card."""
    from src.m8_event_simulator import EventSimResult
    from dataclasses import asdict
    r = result if isinstance(result, dict) else asdict(result)

    intensity = r.get("peak_congestion_intensity", "LOW")
    cascade   = r.get("cascade_risk", "LOW")
    ie = INTENSITY_EMOJI.get(intensity, "⚪")
    ce = CASCADE_EMOJI.get(cascade, "✅")

    closure_lines = "\n".join(
        f"   🚧 {z}" for z in r.get("road_closure_zones", [])
    ) or "   None required"

    diversion_lines = "\n".join(
        f"   ↪️ {d}" for d in r.get("diversion_routes", [])
    )

    notes = r.get("policy_notes", [])
    notes_text = "\n".join(f"   • {n[:80]}{'…' if len(n)>80 else ''}" for n in notes[:2])

    hawkes_alert = r.get("hawkes_alert", "")
    ha_line = f"\n⚡ <b>Hawkes Alert:</b> {hawkes_alert}" if hawkes_alert else ""

    corridors = ", ".join(r.get("affected_corridors", []))
    msg = (
        f"📅 <b>ATLAS EVENT IMPACT REPORT</b>\n"
        f"{'─'*32}\n"
        f"{ie} <b>INTENSITY: {intensity}</b>  "
        f"{ce} Cascade risk: {cascade}\n\n"
        f"🎪 <b>Event:</b> {r.get('event_type','').replace('_',' ').title()}\n"
        f"📍 <b>Venue:</b> {r.get('venue','')}\n"
        f"👥 <b>Expected crowd:</b> {r.get('crowd_size',0):,}\n"
        f"📈 <b>Surge multiplier:</b> {r.get('surge_multiplier',1.0):.1f}× baseline\n\n"
        f"{'─'*32}\n"
        f"🛣 <b>Affected corridors:</b> {corridors}\n"
        f"⏰ <b>Pre-event window:</b> {r.get('pre_event_window','')}\n"
        f"🔴 <b>Peak congestion:</b> {r.get('peak_window','')}\n"
        f"⬇️ <b>Post-event window:</b> {r.get('post_event_window','')}\n\n"
        f"{'─'*32}\n"
        f"👮 <b>Officers required:</b> {r.get('officers_required',0)}\n"
        f"🚧 <b>Barricades required:</b> {r.get('barricades_required',0)}\n"
        f"📦 <b>Extra incidents expected:</b> ~{r.get('expected_extra_events_per_corridor',0):.0f}/corridor\n\n"
        f"<b>Road closure zones:</b>\n{closure_lines}\n\n"
        f"<b>Diversion routes:</b>\n{diversion_lines}"
        f"{ha_line}\n\n"
        f"<b>Policy notes:</b>\n{notes_text}\n\n"
        f"<i>🤖 ATLAS M8 Event Simulator · {datetime.now().strftime('%H:%M IST')}</i>"
    )
    return msg


def format_risk_snapshot() -> str:
    """Quick network risk snapshot from Hawkes + TE data."""
    if not _hawkes:
        return "⚠️ Hawkes model not loaded. Run <code>python run_all.py</code> first."

    tier1 = {k: v for k, v in _hawkes.items()
              if isinstance(v, dict) and v.get("tier") == 1}
    lines = []
    for corr, params in sorted(tier1.items(), key=lambda x: -x[1].get("branching_ratio", 0)):
        n = params.get("branching_ratio", 0)
        emoji = "🔴" if n > 0.2 else "🟡" if n > 0.1 else "🟢"
        lines.append(f"{emoji} <b>{corr}</b>: n={n:.3f}")

    te_sources = []
    if _te_data:
        out_str = _te_data.get("out_strength", {})
        top = sorted(out_str.items(), key=lambda x: -x[1])[:3]
        te_sources = [f"⚡ {k} (TE={v:.4f})" for k, v in top]

    return (
        f"📡 <b>ATLAS NETWORK RISK SNAPSHOT</b>\n"
        f"<i>{datetime.now().strftime('%d %b %Y, %H:%M IST')}</i>\n"
        f"{'─'*32}\n"
        f"<b>Self-Excitation (Hawkes n):</b>\n"
        + "\n".join(lines) +
        f"\n\n<b>Top Contagion Sources (Transfer Entropy):</b>\n"
        + "\n".join(te_sources or ["Data not loaded"]) +
        f"\n\n<i>n = fraction of events that trigger another event on the same corridor</i>"
    )


# ── Command handlers ──────────────────────────────────────────────────────────
@bot.message_handler(commands=["start", "help"])
def handle_help(msg: Message):
    bot.reply_to(msg, (
        "🚦 <b>ATLAS Dispatch Bot</b>\n"
        "Adaptive Traffic Learning &amp; Analysis System\n\n"
        "<b>Just send any incident description!</b>\n"
        "Examples:\n"
        '  • "Truck breakdown on Mysore Road near Kengeri"\n'
        '  • "Heavy rainfall flooding at Silk Board junction"\n'
        '  • "IPL match at Chinnaswamy tomorrow 2pm, 50000 crowd"\n'
        '  • "VIP convoy from Vidhana Soudha at 10am"\n\n'
        "<b>Commands:</b>\n"
        "/risk — Current network risk snapshot\n"
        "/dispatch &lt;message&gt; — Force M6 dispatch query\n"
        "/simulate &lt;message&gt; — Force M8 event simulation\n"
        "/help — Show this message\n\n"
        "<i>Powered by ATLAS M6 ResourceRAG + M8 Event Simulator</i>"
    ))


@bot.message_handler(commands=["risk"])
def handle_risk(msg: Message):
    bot.send_chat_action(msg.chat.id, "typing")
    bot.reply_to(msg, format_risk_snapshot())


def _strip_command(text: str) -> str:
    """Strip /command or /command@BotName prefix from a Telegram message."""
    import re
    return re.sub(r'^/\w+(@\w+)?\s*', '', text).strip()


@bot.message_handler(commands=["dispatch"])
def handle_dispatch_cmd(msg: Message):
    text = _strip_command(msg.text)
    if not text:
        bot.reply_to(msg, "Usage: /dispatch &lt;incident description&gt;")
        return
    _handle_text_core(msg, text, force_mode="dispatch")


@bot.message_handler(commands=["simulate"])
def handle_simulate_cmd(msg: Message):
    text = _strip_command(msg.text)
    if not text:
        bot.reply_to(msg, "Usage: /simulate &lt;planned event description&gt;")
        return
    _handle_text_core(msg, text, force_mode="simulate")


@bot.message_handler(content_types=["text"])
def handle_text(msg: Message):
    _handle_text_core(msg, msg.text)


@bot.message_handler(content_types=["voice"])
def handle_voice(msg: Message):
    """Voice messages — transcribe via file download + fallback message."""
    bot.send_chat_action(msg.chat.id, "typing")
    bot.reply_to(
        msg,
        "🎙️ <b>Voice received.</b>\n\n"
        "Transcription requires Whisper API integration.\n"
        "For the demo, please type the incident description.\n\n"
        "<i>Tip: In production, this integrates with Google Speech-to-Text "
        "or Whisper for full voice dispatch.</i>"
    )


def _handle_text_core(msg: Message, text: str, force_mode: str = None):
    bot.send_chat_action(msg.chat.id, "typing")
    log.info(f"[{msg.chat.id}] Input: {text[:80]}")

    from bot.nlp_parser import parse_incident
    parsed = parse_incident(text)

    mode = force_mode or parsed["mode"]

    if mode == "simulate":
        # Route to M8
        from src.m8_event_simulator import simulate_event
        try:
            result = simulate_event(**parsed["event_dict"])
            response = format_event_card(result)
            bot.reply_to(msg, response)
            log.info(f"[{msg.chat.id}] M8 response sent — {result.event_type}")
        except Exception as e:
            bot.reply_to(msg, f"⚠️ Simulation error: {e}")

    else:
        # Route to M6
        try:
            _ensure_rag()
            from src.m6_resource_rag import query
            card = query(parsed["event_dict"], k=5)

            # Attach TE downstream and hawkes alert into extracted for radio script
            extracted = parsed["extracted"].copy()
            if _hawkes and extracted.get("corridor") in _hawkes:
                n = _hawkes[extracted["corridor"]].get("branching_ratio", 0)
                if n > 0.15:
                    extracted["hawkes_alert"] = (
                        f"n={n:.3f} — high self-excitation. "
                        f"Secondary breakdown within 30 min likely."
                    )
            if _te_data:
                sig = _te_data.get("significant_edges", [])
                extracted["te_downstream"] = [
                    e["dst"] for e in sig if e["src"] == extracted.get("corridor", "")
                ][:2]
            card["cascade_risk"] = (
                "HIGH" if extracted.get("hawkes_alert") else "LOW"
            )

            response = format_dispatch_card(card, extracted)
            bot.reply_to(msg, response)
            log.info(f"[{msg.chat.id}] M6 dispatch sent — conf={card['confidence_score']:.2f}")

            # ── GenAI Radio Script (last-mile translation) ────────────────────
            try:
                from src.radio_dispatch import generate_radio_script
                radio = generate_radio_script(card, extracted)
                src_tag = "🤖 Gemini 1.5 Flash" if "gemini" in radio["source"] else "📋 Template"
                radio_msg = (
                    f"📻 <b>ATLAS RADIO SCRIPT</b>  <i>({src_tag})</i>\n"
                    f"{'─'*32}\n"
                    f"🇬🇧 <b>English:</b>\n"
                    f"<code>{radio['english']}</code>\n\n"
                    f"🇮🇳 <b>ಕನ್ನಡ (Kannada):</b>\n"
                    f"<code>{radio['kannada']}</code>"
                )
                bot.reply_to(msg, radio_msg)
                log.info(f"[{msg.chat.id}] Radio script sent — source={radio['source']}")
            except Exception as re:
                log.warning(f"Radio script skipped: {re}")

        except Exception as e:
            bot.reply_to(msg, f"⚠️ Dispatch error: {e}\n\n"
                             "Ensure models are built: <code>python run_all.py</code>")



# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Loading ATLAS models...")
    _load_models()
    _ensure_rag()
    log.info("✅ ATLAS Dispatch Bot online. Polling Telegram...")
    log.info("   Send a message to test the bot.")
    bot.infinity_polling(timeout=30, long_polling_timeout=20)
