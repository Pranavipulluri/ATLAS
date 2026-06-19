"""
M5 — Extreme Value Theory (EVT)
GPD fitting on chronic event resolution time tails.
Outputs: models/evt_results.json, figures/evt_*.png
"""

import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
from scipy.stats import genpareto

warnings.filterwarnings('ignore')

DATA_DIR  = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"
FIG_DIR   = Path(__file__).parent.parent / "figures"
MODEL_DIR.mkdir(exist_ok=True)

INFRA_GROUPS = {
    'road_surface': ['pot_holes', 'road_conditions'],
    'drainage':     ['water_logging'],
    'construction': ['construction'],
    'vegetation':   ['tree_fall'],
}


def mean_excess_threshold(data: np.ndarray, pct_range=(50, 92)) -> float:
    """
    Select threshold where mean excess plot becomes approximately linear.
    Returns threshold at 85th percentile as safe default.
    """
    thresholds = np.percentile(data, np.arange(*pct_range, 5))
    me_vals = []
    for u in thresholds:
        exc = data[data > u] - u
        if len(exc) >= 10:
            me_vals.append((u, exc.mean(), len(exc)))
    # Use 85th percentile as default (balance between N and tail focus)
    return float(np.percentile(data, 85))


def fit_gpd(data: np.ndarray, threshold_pct: float = 85) -> dict | None:
    """Fit Generalized Pareto Distribution to exceedances above threshold."""
    u = float(np.percentile(data, threshold_pct))
    exceedances = data[data > u] - u
    N_u = len(exceedances)
    N_total = len(data)

    if N_u < 15:
        return None  # Insufficient tail data

    try:
        xi, loc, sigma = genpareto.fit(exceedances, floc=0)
    except Exception:
        return None

    # Return level: z_p = u + (σ/ξ)[(n/(N_u·p))^ξ - 1]
    def return_level(return_period_days: float, events_per_day: float = 1.0) -> float:
        p = 1 / (return_period_days * events_per_day)
        rate = N_total / N_u
        if abs(xi) < 1e-6:  # ξ ≈ 0: exponential
            return u + sigma * np.log(rate / p)
        val = u + (sigma / xi) * ((rate / p) ** xi - 1)
        return float(max(val, u))

    # Confidence interval via bootstrap
    ci_lo, ci_hi = _bootstrap_ci(exceedances, return_period_days=30)

    tail_type = 'heavy' if xi > 0.1 else 'exponential' if abs(xi) < 0.1 else 'bounded'

    return {
        'threshold_u_mins':    float(u),
        'threshold_u_days':    float(u / 1440),
        'N_total':             N_total,
        'N_exceedances':       N_u,
        'xi':                  float(xi),
        'sigma':               float(sigma),
        'tail_type':           tail_type,
        'return_30d_mins':     return_level(30),
        'return_30d_days':     return_level(30) / 1440,
        'return_90d_mins':     return_level(90),
        'return_90d_days':     return_level(90) / 1440,
        'return_180d_mins':    return_level(180),
        'return_180d_days':    return_level(180) / 1440,
        'ci_30d_lo_days':      ci_lo,
        'ci_30d_hi_days':      ci_hi,
        'policy_implication':  _policy_text(tail_type, xi, return_level(30) / 1440),
    }


def _bootstrap_ci(exceedances: np.ndarray, return_period_days: float = 30,
                  n_boot: int = 200, seed: int = 42) -> tuple:
    """Bootstrap 95% CI for return level at given period."""
    rng = np.random.default_rng(seed)
    boot_levels = []
    for _ in range(n_boot):
        sample = rng.choice(exceedances, size=len(exceedances), replace=True)
        try:
            xi_b, _, sig_b = genpareto.fit(sample, floc=0)
            p = 1 / return_period_days
            if abs(xi_b) < 1e-6:
                level = sig_b * np.log(1 / p)
            else:
                level = (sig_b / xi_b) * ((1 / p) ** xi_b - 1)
            boot_levels.append(level / 1440)  # in days
        except Exception:
            continue
    if not boot_levels:
        return 0.0, 0.0
    return float(np.percentile(boot_levels, 2.5)), float(np.percentile(boot_levels, 97.5))


def _policy_text(tail_type, xi, rl_30_days):
    if tail_type == 'heavy':
        return (f"HEAVY TAIL (ξ={xi:.3f}): Worst-case duration is unbounded. "
                f"Plan for at least {rl_30_days:.0f} days of traffic diversion for a single event "
                f"occurring once per month. Mean-based planning severely underestimates requirements.")
    elif tail_type == 'exponential':
        return (f"EXPONENTIAL TAIL (ξ≈0): Duration is bounded but variable. "
                f"Plan for {rl_30_days:.0f} days worst-case per monthly event.")
    else:
        return (f"BOUNDED TAIL (ξ={xi:.3f}): Worst-case duration is mathematically capped. "
                f"Standard planning with {rl_30_days:.0f} day buffer is adequate.")


def run_evt(df: pd.DataFrame) -> dict:
    print("[M5] Running Extreme Value Theory...")
    df_chronic = df[df['infrastructure_group'] != 'N/A'].copy()
    df_chronic = df_chronic[df_chronic['resolution_valid']].dropna(subset=['resolution_mins'])

    all_results = {}

    for group, causes in INFRA_GROUPS.items():
        subset = df_chronic[df_chronic['event_cause_clean'].isin(causes)]['resolution_mins'].values
        print(f"\n[M5]   Group '{group}' (causes: {causes}): N={len(subset)}")

        if len(subset) < 20:
            print(f"[M5]   SKIP: N < 20")
            all_results[group] = {'error': 'insufficient_data', 'N': len(subset)}
            continue

        result = fit_gpd(subset)
        if result is None:
            print(f"[M5]   SKIP: GPD fit failed")
            all_results[group] = {'error': 'fit_failed', 'N': len(subset)}
            continue

        result['group'] = group
        result['causes'] = causes
        result['data_summary'] = {
            'median_mins': float(np.median(subset)),
            'median_days': float(np.median(subset) / 1440),
            'p95_days': float(np.percentile(subset, 95) / 1440),
            'max_days': float(np.max(subset) / 1440),
        }

        all_results[group] = result
        print(f"[M5]   ξ={result['xi']:.3f} ({result['tail_type']} tail)")
        print(f"[M5]   30-day return level: {result['return_30d_days']:.1f} days")
        print(f"[M5]   90-day return level: {result['return_90d_days']:.1f} days")
        print(f"[M5]   Policy: {result['policy_implication'][:80]}...")

    # Save
    out = MODEL_DIR / "evt_results.json"
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[M5] ✅ Saved → {out}")

    _plot_return_levels(all_results, df_chronic, FIG_DIR / "evt_return_levels.png")
    _plot_qq(all_results, df_chronic, FIG_DIR / "evt_qq.png")

    return all_results


def _plot_return_levels(results, df_chronic, out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#1a1d2e')

        colors = {'road_surface': '#ff6b6b', 'drainage': '#00d4ff',
                  'construction': '#ffd93d', 'vegetation': '#a855f7'}

        periods = np.logspace(0, 2.5, 50)  # 1 to ~300 days

        for group, res in results.items():
            if 'error' in res or res.get('N', 0) < 20:
                continue
            xi, sigma = res['xi'], res['sigma']
            u = res['threshold_u_mins']
            N_total = res['N_total']
            N_u = res['N_exceedances']

            levels = []
            for T in periods:
                p = 1 / T
                rate = N_total / N_u
                if abs(xi) < 1e-6:
                    lev = u + sigma * np.log(rate / p)
                else:
                    lev = u + (sigma / xi) * ((rate / p) ** xi - 1)
                levels.append(max(lev, u) / 1440)  # days

            ax.plot(periods, levels, color=colors.get(group, 'white'),
                    lw=2.5, label=group)
            # CI band
            lo = res.get('ci_30d_lo_days', levels[20] * 0.7)
            hi = res.get('ci_30d_hi_days', levels[20] * 1.4)
            ax.fill_between([28, 32], lo, hi, alpha=0.3, color=colors.get(group, 'white'))

        ax.set_xscale('log')
        ax.set_xlabel('Return period (days)', color='white')
        ax.set_ylabel('Expected worst-case duration (days)', color='white')
        ax.set_title('EVT Return Level Curves — Chronic Event Planning Buffers\n(shaded = 95% CI at 30-day return period)',
                     color='white', fontsize=12, fontweight='bold')
        ax.tick_params(colors='white')
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
        for sp in ['bottom', 'left']: ax.spines[sp].set_color('#555')
        ax.legend(framealpha=0.3, labelcolor='white', fontsize=10)
        ax.grid(axis='y', color='#333', alpha=0.5)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, facecolor='#0f1117')
        plt.close()
        print(f"[M5]   Plot saved → {out_path}")
    except Exception as e:
        print(f"[M5]   Plot skipped: {e}")


def _plot_qq(results, df_chronic, out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        groups_with_results = [g for g, r in results.items() if 'xi' in r]
        if not groups_with_results:
            return

        n_plots = len(groups_with_results)
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]
        fig.patch.set_facecolor('#0f1117')

        for ax, group in zip(axes, groups_with_results):
            ax.set_facecolor('#1a1d2e')
            res = results[group]
            causes = INFRA_GROUPS[group]
            data = df_chronic[df_chronic['event_cause_clean'].isin(causes)]['resolution_mins'].values
            u = res['threshold_u_mins']
            exceedances = np.sort(data[data > u] - u)
            if len(exceedances) < 5:
                continue
            xi, sigma = res['xi'], res['sigma']
            theoretical = genpareto.ppf(
                np.linspace(0.01, 0.99, len(exceedances)), xi, scale=sigma
            )
            ax.scatter(theoretical / 1440, exceedances / 1440,
                       alpha=0.6, color='#00d4ff', s=20, edgecolors='none')
            lim = max(theoretical.max(), exceedances.max()) / 1440 * 1.1
            ax.plot([0, lim], [0, lim], 'r--', lw=1.5, label='Perfect fit')
            ax.set_xlabel('Theoretical (days)', color='white', fontsize=9)
            ax.set_ylabel('Empirical (days)', color='white', fontsize=9)
            ax.set_title(f'{group}\nξ={xi:.3f} ({res["tail_type"]})', color='white', fontsize=10)
            ax.tick_params(colors='white', labelsize=7)
            for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
            for sp in ['bottom', 'left']: ax.spines[sp].set_color('#555')

        fig.suptitle('EVT QQ Plots — GPD Goodness of Fit', color='white',
                     fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, facecolor='#0f1117', bbox_inches='tight')
        plt.close()
        print(f"[M5]   Plot saved → {out_path}")
    except Exception as e:
        print(f"[M5]   Plot skipped: {e}")


if __name__ == "__main__":
    df = pd.read_parquet(DATA_DIR / "astram_clean.parquet")
    run_evt(df)
