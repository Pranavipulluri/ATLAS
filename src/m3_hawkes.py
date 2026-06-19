"""
M3 — Spatiotemporal Hawkes Process
KS pre-test + dual-background MLE fit per corridor + GoF + forecasting.
Outputs: models/hawkes_results.json, figures/hawkes_*.png
"""

import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import kstest, expon

warnings.filterwarnings('ignore')

DATA_DIR  = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"
FIG_DIR   = Path(__file__).parent.parent / "figures"
MODEL_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

TIER1 = ['Mysore Road', 'Bellary Road 1', 'Tumkur Road', 'Bellary Road 2']
TIER2 = ['Hosur Road', 'ORR North 1', 'Old Madras Road', 'Magadi Road', 'ORR East 1',
         'ORR North 2', 'West of Chord Road', 'ORR West 1', 'Bannerghatta Road']

ZONE_FOR_CORRIDOR = {
    'Hosur Road': 'South Zone 1', 'ORR North 1': 'North Zone 1',
    'Old Madras Road': 'East Zone 1', 'Magadi Road': 'West Zone 2',
    'ORR East 1': 'East Zone 2', 'ORR North 2': 'North Zone 2',
    'West of Chord Road': 'West Zone 1', 'ORR West 1': 'West Zone 1',
    'Bannerghatta Road': 'South Zone 1',
}


def get_timestamps(df, corridor, cause='vehicle_breakdown'):
    """Extract sorted timestamps (minutes since epoch) + hour-of-day array + t0 offset."""
    sub = df[
        (df['corridor'] == corridor) &
        (df['event_cause_clean'] == cause) &
        df['start_datetime_IST'].notna()
    ].sort_values('start_datetime_IST')
    if len(sub) < 30:
        return np.array([]), np.array([]), 0.0
    # Convert to minutes since first event
    t0 = sub['start_datetime_IST'].iloc[0]
    ts = sub['start_datetime_IST'].apply(
        lambda t: (t - t0).total_seconds() / 60
    ).values
    hours = sub['hour_IST'].values.astype(float)
    t0_offset_mins = float(t0.hour * 60 + t0.minute + t0.second / 60)
    return ts, hours, t0_offset_mins


def ks_test_poisson(ts):
    """KS test: do inter-arrivals follow Exp (Poisson null)?"""
    ia = np.diff(ts)
    ia = ia[ia > 0]
    if len(ia) < 20:
        return None, None
    scale = ia.mean()
    ks_stat, p_val = kstest(ia, 'expon', args=(0, scale))
    return float(ks_stat), float(p_val)


def cv_statistic(ts):
    """Coefficient of variation of inter-arrivals. CV > 1 → clustering."""
    ia = np.diff(ts)
    ia = ia[ia > 0]
    if len(ia) < 5:
        return None
    return float(ia.std() / ia.mean())


def compute_background_integral(timestamps_minutes, mu_day, mu_night, t0_offset_mins=0.0,
                                 night_start_h=20, night_end_h=6):
    """Sum mu over observation window, switching between day/night rate."""
    total = 0.0
    all_points = [0.0] + list(timestamps_minutes)
    for i in range(len(all_points) - 1):
        t_mid = (all_points[i] + all_points[i+1]) / 2.0
        hour = ((t_mid + t0_offset_mins) % 1440.0) / 60.0
        dt = all_points[i+1] - all_points[i]
        mu = mu_night if (hour >= night_start_h or hour < night_end_h) else mu_day
        total += mu * dt
    return total


def hawkes_nll(params, timestamps, hours_of_day, t0_offset_mins=0.0):
    """
    Negative log-likelihood of dual-background Hawkes process.
    params = [mu_day, mu_night, alpha, beta]

    Exact formula (Ozaki 1979):
        log L = Σᵢ log λ(tᵢ) − Λ(T)
    where:
        λ(tᵢ) = μ(tᵢ) + α · Aᵢ
        Aᵢ     = (Aᵢ₋₁ + 1) · exp(−β·(tᵢ − tᵢ₋₁))    [A₀ = 0]
        Λ(T)   = μ_day·T_day + μ_night·T_night + (α/β)·Σᵢ(1−exp(−β(T−tᵢ)))
    """
    mu_day, mu_night, alpha, beta = params
    if mu_day <= 0 or mu_night <= 0 or alpha <= 0 or beta <= 0:
        return 1e10
    if alpha >= beta:          # stationarity: branching ratio n = α/β < 1
        return 1e10

    n = len(timestamps)
    T = timestamps[-1]

    # ── Log-likelihood sum ─────────────────────────────────────────────────
    ll = 0.0
    A  = 0.0   # A₀ = 0: no prior events before t₀

    for i in range(n):
        # At i=0: A=0 (no excitation from previous events)
        # At i>0: A decays the PREVIOUS running sum (including event i-1)
        # Correct Ozaki recursion: A[i] = (A[i-1] + 1) * exp(-β * (t[i] - t[i-1]))
        if i > 0:
            A = (A + 1.0) * np.exp(-beta * (timestamps[i] - timestamps[i - 1]))

        mu_i  = mu_night if hours_of_day[i] >= 20 or hours_of_day[i] < 6 else mu_day
        lam_i = mu_i + alpha * A
        if lam_i <= 1e-12:
            return 1e10
        ll += np.log(lam_i)

    # ── Compensator Λ(T) ──────────────────────────────────────────────────
    comp_mu  = compute_background_integral(timestamps, mu_day, mu_night, t0_offset_mins)

    # Excitation integral: (α/β) · Σᵢ (1 − exp(−β(T − tᵢ)))
    comp_exc = (alpha / beta) * float(np.sum(1.0 - np.exp(-beta * (T - timestamps))))

    return -(ll - (comp_mu + comp_exc))



def fit_hawkes(ts, hours, t0_offset=0.0, n_restarts=12, seed=0):
    """Fit Hawkes with multiple random restarts."""
    rng = np.random.default_rng(seed)
    best = None

    # Mean IAT for scaling initial guesses
    diffs = np.diff(ts[ts > 0])
    mean_iat = float(diffs.mean()) if len(diffs) > 0 else 60.0
    mu_scale = 1.0 / (mean_iat + 1.0)  # expected background rate

    for restart_i in range(n_restarts):
        # Wide random restarts to escape local minima
        x0 = [
            rng.uniform(0.3 * mu_scale, 3.0 * mu_scale),   # mu_day
            rng.uniform(0.5 * mu_scale, 5.0 * mu_scale),   # mu_night (higher)
            rng.uniform(0.05, 0.8),                         # alpha
            rng.uniform(0.1, 3.0),                          # beta  (alpha/beta=branching ratio)
        ]
        try:
            res = minimize(
                hawkes_nll, x0, args=(ts, hours, t0_offset),
                method='L-BFGS-B',
                bounds=[(1e-8, 5), (1e-8, 5), (1e-6, 0.999), (1e-4, 20)],
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8},
            )
            if res.fun < 1e9:
                if best is None or res.fun < best.fun:
                    best = res
        except Exception:
            continue

    if best is None:
        return None
    mu_day, mu_night, alpha, beta = best.x
    branching_ratio = alpha / beta
    return {
        'mu_day':   float(mu_day),
        'mu_night': float(mu_night),
        'alpha':    float(alpha),
        'beta':     float(beta),
        'branching_ratio':              float(branching_ratio),
        'mean_excitation_decay_mins':   float(1 / beta),
        'nll':      float(best.fun),
        'stationary': bool(branching_ratio < 1),
    }


def gof_test(ts, params, t0_offset=0.0):
    """Goodness-of-fit: compensator residuals should be Exp(1)."""
    mu_day = params['mu_day']
    mu_night = params['mu_night']
    alpha = params['alpha']
    beta = params['beta']
    n = len(ts)
    compensators = []
    A = 0.0
    for i in range(1, n):
        dt = ts[i] - ts[i - 1]
        hour = ((ts[i-1] + t0_offset) % 1440.0) / 60.0
        mu = mu_night if (hour >= 20 or hour < 6) else mu_day
        comp = mu * dt + (alpha / beta) * A * (1 - np.exp(-beta * dt))
        compensators.append(comp)
        A = (A + 1.0) * np.exp(-beta * dt)   # Ozaki recursion: matches NLL order
    compensators = np.array(compensators)
    compensators = compensators[compensators > 0]
    if len(compensators) < 10:
        return None, None
    ks, p = kstest(compensators, 'expon', args=(0, 1))
    return float(ks), float(p)


def forecast_intensity(params, recent_ts_mins, horizon_mins=[60, 180, 360]):
    """Expected event counts in next horizon given recent events."""
    alpha = params['alpha']
    beta = params['beta']
    mu = params['mu_night']  # conservative (night = higher)
    T_now = recent_ts_mins[-1] if len(recent_ts_mins) > 0 else 0.0

    results = {}
    for h in horizon_mins:
        excitation = sum(
            alpha * np.exp(-beta * (T_now + h - ti))
            for ti in recent_ts_mins
            if ti <= T_now
        )
        lam = mu + excitation
        exp_count = lam * h
        results[f'{h}min'] = {
            'lambda': round(lam, 5),
            'expected_events': round(exp_count, 2),
            'alert': 'HIGH' if exp_count > 2 else 'MEDIUM' if exp_count > 0.5 else 'LOW',
        }
    return results


def run_hawkes(df: pd.DataFrame) -> dict:
    print("[M3] Running Hawkes Process...")
    all_results = {}

    # ── KS pre-tests ─────────────────────────────────────────────────────────
    print("[M3] KS pre-tests (Poisson null):")
    ks_results = {}
    for corr in TIER1 + TIER2[:5]:
        ts, hours, _ = get_timestamps(df, corr)
        if len(ts) < 30:
            continue
        ks, p = ks_test_poisson(ts)
        cv = cv_statistic(ts)
        reject = (p is not None and p < 0.05)
        ks_results[corr] = {'ks': ks, 'p': p, 'cv': cv, 'reject_poisson': reject,
                             'n': len(ts)}
        marker = 'REJECT' if reject else '– keep'
        print(f"  {corr:<25} N={len(ts):3d}  CV={cv:.3f}  p={p:.4f}  {marker}")

    # ── Tier-1: individual corridor fits ─────────────────────────────────────
    print("\n[M3] Fitting Tier-1 corridors (individual):")
    for corr in TIER1:
        ts, hours, t0_offset = get_timestamps(df, corr)
        if len(ts) < 40:
            print(f"  {corr}: SKIP (N={len(ts)} < 40)")
            continue
        params = fit_hawkes(ts, hours, t0_offset)
        if params is None:
            print(f"  {corr}: FIT FAILED")
            continue
        ks_gof, p_gof = gof_test(ts, params, t0_offset)
        params['N'] = len(ts)
        params['ks_pretest'] = ks_results.get(corr, {})
        params['gof_ks'] = ks_gof
        params['gof_p'] = p_gof
        params['tier'] = 1
        params['corridor'] = corr
        all_results[corr] = params
        print(f"  {corr:<25} n={params['branching_ratio']:.3f}  "
              f"μ_night={params['mu_night']:.5f}/min  "
              f"decay={params['mean_excitation_decay_mins']:.0f}min  "
              f"GoF_p={p_gof:.3f}")

    # ── Tier-2: zone-pooled fits ──────────────────────────────────────────────
    print("\n[M3] Fitting Tier-2 (zone-pooled):")
    zone_data = {}  # zone -> (ts_list, hours_list)
    for corr in TIER2:
        zone = ZONE_FOR_CORRIDOR.get(corr, 'zone_unknown')
        ts, hours, _ = get_timestamps(df, corr)
        if len(ts) < 5:
            continue
        if zone not in zone_data:
            zone_data[zone] = ([], [])
        zone_data[zone][0].extend(ts.tolist())
        zone_data[zone][1].extend(hours.tolist())

    for zone, (ts_list, hrs_list) in zone_data.items():
        if len(ts_list) < 30:
            continue
        sort_idx = np.argsort(ts_list)
        ts    = np.array(ts_list)[sort_idx]
        hours = np.array(hrs_list)[sort_idx]
        params = fit_hawkes(ts, hours, t0_offset=0.0)
        if params is None:
            continue
        params['N'] = len(ts)
        params['tier'] = 2
        params['zone'] = zone
        all_results[f'ZONE:{zone}'] = params
        print(f"  ZONE:{zone:<25} n={params['branching_ratio']:.3f}  "
              f"μ_night={params['mu_night']:.5f}/min")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n[M3] Branching ratio summary:")
    tier1_ratios = {k: v['branching_ratio'] for k, v in all_results.items() if v.get('tier') == 1}
    for corr, n in sorted(tier1_ratios.items(), key=lambda x: -x[1]):
        interp = 'STRONG self-excitation' if n > 0.3 else 'MODERATE' if n > 0.1 else 'WEAK'
        print(f"  {corr:<25} n={n:.3f}  → {interp}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out = MODEL_DIR / "hawkes_results.json"
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[M3] ✅ Saved → {out}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_branching_ratios(all_results, FIG_DIR / "hawkes_branching.png")
    _plot_excitation_kernels(all_results, FIG_DIR / "hawkes_kernels.png")
    _plot_hourly_intensity(df, FIG_DIR / "hawkes_hourly.png")

    return all_results


def _plot_branching_ratios(results, out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        tier1 = {k: v for k, v in results.items() if v.get('tier') == 1}
        if not tier1:
            return
        corridors = list(tier1.keys())
        ratios = [tier1[c]['branching_ratio'] for c in corridors]

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#1a1d2e')
        colors = ['#ff6b6b' if r > 0.3 else '#ffd93d' if r > 0.1 else '#00d4ff' for r in ratios]
        bars = ax.barh(corridors, ratios, color=colors, edgecolor='none', height=0.6)
        ax.axvline(0.3, color='#ff6b6b', lw=1.5, linestyle='--', label='Strong excitation (n=0.3)')
        ax.axvline(0.1, color='#ffd93d', lw=1.5, linestyle='--', label='Moderate excitation (n=0.1)')
        ax.set_xlabel('Branching Ratio (n = α/β)', color='white')
        ax.set_title('Hawkes Branching Ratios by Corridor\n(fraction of events caused by prior events)',
                     color='white', fontsize=12, fontweight='bold')
        ax.tick_params(colors='white')
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
        for sp in ['bottom', 'left']: ax.spines[sp].set_color('#555')
        for bar, ratio in zip(bars, ratios):
            ax.text(ratio + 0.005, bar.get_y() + bar.get_height() / 2,
                    f'{ratio:.3f}', va='center', color='white', fontsize=9)
        ax.legend(framealpha=0.3, labelcolor='white', fontsize=9)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, facecolor='#0f1117')
        plt.close()
        print(f"[M3]   Plot saved → {out_path}")
    except Exception as e:
        print(f"[M3]   Plot skipped: {e}")


def _plot_excitation_kernels(results, out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        tier1 = {k: v for k, v in results.items() if v.get('tier') == 1}
        if not tier1:
            return
        t = np.linspace(0, 300, 500)  # 0 to 5 hours in minutes
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#1a1d2e')
        palette = ['#00d4ff', '#ff6b6b', '#ffd93d', '#a855f7', '#10b981']
        for i, (corr, params) in enumerate(tier1.items()):
            kernel = params['alpha'] * np.exp(-params['beta'] * t)
            ax.plot(t, kernel, color=palette[i % len(palette)], lw=2.5,
                    label=f"{corr} (n={params['branching_ratio']:.3f})")
        ax.set_xlabel('Time since triggering event (minutes)', color='white')
        ax.set_ylabel('Excitation intensity α·e^(−βt)', color='white')
        ax.set_title('Hawkes Excitation Kernels — How fast excitation decays',
                     color='white', fontsize=12, fontweight='bold')
        ax.tick_params(colors='white')
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
        for sp in ['bottom', 'left']: ax.spines[sp].set_color('#555')
        ax.legend(framealpha=0.3, labelcolor='white', fontsize=9)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, facecolor='#0f1117')
        plt.close()
        print(f"[M3]   Plot saved → {out_path}")
    except Exception as e:
        print(f"[M3]   Plot skipped: {e}")


def _plot_hourly_intensity(df, out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        hourly = df.groupby('hour_IST').size().reindex(range(24), fill_value=0)
        hourly_bd = df[df['event_cause_clean'] == 'vehicle_breakdown'].groupby('hour_IST').size().reindex(range(24), fill_value=0)

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#1a1d2e')
        x = np.arange(24)
        ax.bar(x, hourly.values, color='#3d4166', edgecolor='none', label='All events')
        ax.bar(x, hourly_bd.values, color='#00d4ff', edgecolor='none', alpha=0.9, label='Breakdowns')
        ax.axvspan(20, 24, alpha=0.15, color='#ff6b6b', label='Night window (8pm–6am)')
        ax.axvspan(0, 6, alpha=0.15, color='#ff6b6b')
        ax.set_xlabel('Hour (IST)', color='white')
        ax.set_ylabel('Event count', color='white')
        ax.set_title('Hourly Event Distribution — IST\n(dual-background regime for Hawkes)',
                     color='white', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{h:02d}h' for h in x], rotation=45, fontsize=8)
        ax.tick_params(colors='white')
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
        for sp in ['bottom', 'left']: ax.spines[sp].set_color('#555')
        ax.legend(framealpha=0.3, labelcolor='white', fontsize=9)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, facecolor='#0f1117')
        plt.close()
        print(f"[M3]   Plot saved → {out_path}")
    except Exception as e:
        print(f"[M3]   Plot skipped: {e}")


if __name__ == "__main__":
    df = pd.read_parquet(DATA_DIR / "astram_clean.parquet")
    run_hawkes(df)
