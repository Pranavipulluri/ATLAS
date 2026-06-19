"""
run_all.py — Master runner for ATLAS system
Executes all 8 models in sequence and prepares the dashboard.
Usage: python run_all.py
"""

import sys
import time
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def main():
    t_total = time.time()

    # ── M0: Data Pipeline ────────────────────────────────────────────────────
    section("M0: Data Pipeline")
    from src.m0_pipeline import load_and_clean
    df = load_and_clean()

    # ── M1: EM Mixture Model ─────────────────────────────────────────────────
    section("M1: EM Mixture Model")
    from src.m1_em_mixture import run_em
    em_results = run_em(df)

    # ── M2: CauseGraph ───────────────────────────────────────────────────────
    section("M2: CauseGraph")
    from src.m2_cause_graph import run_cause_graph
    cg_results = run_cause_graph(df)

    # ── M3: Hawkes Process ───────────────────────────────────────────────────
    section("M3: Hawkes Process")
    from src.m3_hawkes import run_hawkes
    hawkes_results = run_hawkes(df)

    # ── M4: Transfer Entropy ─────────────────────────────────────────────────
    section("M4: Transfer Entropy")
    from src.m4_transfer_entropy import run_transfer_entropy
    te_results = run_transfer_entropy(df)

    # ── M5: EVT ──────────────────────────────────────────────────────────────
    section("M5: Extreme Value Theory")
    from src.m5_evt import run_evt
    evt_results = run_evt(df)

    # ── M6: ResourceRAG ──────────────────────────────────────────────────────
    section("M6: ResourceRAG")
    from src.m6_resource_rag import run_resource_rag
    rag_results = run_resource_rag(df)

    # ── M7: PostEventCalibrator ───────────────────────────────────────────────
    section("M7: PostEventCalibrator")
    from src.m7_calibrator import run_calibrator
    calib_results = run_calibrator(df)

    # ── M8: Event Impact Simulator ────────────────────────────────────────────
    section("M8: Event Impact Simulator")
    from src.m8_event_simulator import run_event_simulator
    sim_results = run_event_simulator(df)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    section(f"✅ All 8 models complete in {elapsed:.1f}s")
    print(f"  Data:       {ROOT / 'data'}")
    print(f"  Models:     {ROOT / 'models'}")
    print(f"  Figures:    {ROOT / 'figures'}")
    print(f"\n  To launch dashboard:")
    print(f"    python -m streamlit run dashboard/app.py")
    print(f"  To run Telegram bot:")
    print(f"    python bot/dispatch_bot.py")

if __name__ == "__main__":
    main()
