"""
M1 — EM Mixture Model
Proves the acute/chronic bimodal split in resolution times.
Outputs: models/em_params.json, figures/em_bimodal.png
"""

import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
from scipy.special import logsumexp
from scipy.stats import norm as norm_dist

warnings.filterwarnings('ignore')

DATA_DIR   = Path(__file__).parent.parent / "data"
MODEL_DIR  = Path(__file__).parent.parent / "models"
FIG_DIR    = Path(__file__).parent.parent / "figures"
MODEL_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

ACUTE_CAUSES   = {'vehicle_breakdown', 'accident', 'congestion', 'procession', 'protest'}
CHRONIC_CAUSES = {'pot_holes', 'road_conditions', 'water_logging', 'construction', 'tree_fall'}


def em_fit(X: np.ndarray, n_components: int = 2, max_iter: int = 500, tol: float = 1e-8, seed: int = 42):
    """Fit Gaussian mixture in log-space via EM."""
    rng = np.random.default_rng(seed)
    N = len(X)

    # Initialise: acute ≈ log(41min)=3.7, chronic ≈ log(13000min)=9.5
    if n_components == 2:
        mu    = np.array([3.7, 9.5])
        sigma = np.array([1.2, 1.8])
        pi    = np.array([0.65, 0.35])
    else:  # 3 components
        mu    = np.array([3.7, 7.5, 9.8])
        sigma = np.array([1.0, 1.2, 1.5])
        pi    = np.array([0.60, 0.20, 0.20])

    log_likelihoods = []

    for iteration in range(max_iter):
        # ── E-step ──────────────────────────────────────────────────────────
        log_resp = np.column_stack([
            np.log(pi[k] + 1e-300) + norm_dist.logpdf(X, mu[k], sigma[k])
            for k in range(n_components)
        ])
        log_norm = logsumexp(log_resp, axis=1, keepdims=True)
        log_resp -= log_norm
        resp = np.exp(log_resp)

        ll = float(log_norm.sum())
        log_likelihoods.append(ll)

        # ── M-step ──────────────────────────────────────────────────────────
        Nk = resp.sum(axis=0).clip(min=1e-6)
        pi    = Nk / N
        mu    = (resp * X[:, None]).sum(axis=0) / Nk
        sigma = np.sqrt(((resp * (X[:, None] - mu) ** 2).sum(axis=0)) / Nk).clip(min=0.01)

        if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break

    return mu, sigma, pi, resp, log_likelihoods


def bic_score(X: np.ndarray, n_components: int, mu, sigma, pi) -> float:
    N = len(X)
    n_params = n_components * 3 - 1  # means + stds + (weights - 1)
    log_lik = np.zeros(N)
    for k in range(n_components):
        log_lik += pi[k] * norm_dist.pdf(X, mu[k], sigma[k])
    ll = np.log(log_lik + 1e-300).sum()
    return -2 * ll + n_params * np.log(N)


def run_em(df: pd.DataFrame) -> dict:
    print("[M1] Running EM Mixture Model...")
    df_res = df[df['resolution_valid']].dropna(subset=['log_resolution_mins'])
    X = df_res['log_resolution_mins'].values
    print(f"[M1]   N (valid resolution events): {len(X)}")

    # ── Fit 1 and 2 component models ────────────────────────────────────────
    mu1 = np.array([X.mean()])
    sig1 = np.array([X.std()])
    pi1 = np.array([1.0])
    bic1 = bic_score(X, 1, mu1, sig1, pi1)

    mu2, sig2, pi2, resp2, ll_hist2 = em_fit(X, n_components=2)
    bic2 = bic_score(X, 2, mu2, sig2, pi2)

    delta_bic = bic1 - bic2
    print(f"[M1]   BIC (1-component): {bic1:.1f}")
    print(f"[M1]   BIC (2-component): {bic2:.1f}")
    print(f"[M1]   ΔBIC = {delta_bic:.1f}  → {'STRONG evidence for 2 components (>10)' if delta_bic > 10 else 'WEAK'}")

    # ── Cluster assignment ───────────────────────────────────────────────────
    # Component 0 = smaller mu = acute; Component 1 = larger mu = chronic
    acute_idx   = int(np.argmin(mu2))
    chronic_idx = int(np.argmax(mu2))

    df_res = df_res.copy()
    df_res['em_component']  = resp2.argmax(axis=1)
    df_res['em_prob_acute'] = resp2[:, acute_idx]

    # ── Cluster purity ───────────────────────────────────────────────────────
    acute_events   = df_res[df_res['em_component'] == acute_idx]
    chronic_events = df_res[df_res['em_component'] == chronic_idx]
    acute_purity   = acute_events['event_cause_clean'].isin(ACUTE_CAUSES).mean()
    chronic_purity = chronic_events['event_cause_clean'].isin(CHRONIC_CAUSES).mean()
    print(f"[M1]   Acute cluster purity:   {acute_purity:.1%}")
    print(f"[M1]   Chronic cluster purity: {chronic_purity:.1%}")

    # If purity is too low, try 3 components
    if chronic_purity < 0.60:
        print("[M1]   Purity low — trying 3 components...")
        mu3, sig3, pi3, resp3, _ = em_fit(X, n_components=3)
        bic3 = bic_score(X, 3, mu3, sig3, pi3)
        if bic2 - bic3 > 5:
            mu2, sig2, pi2, resp2 = mu3, sig3, pi3, resp3
            bic2 = bic3
            print(f"[M1]   Upgraded to 3 components (BIC={bic3:.1f})")

    # ── Results ─────────────────────────────────────────────────────────────
    results = {
        'n_events':        len(X),
        'bic_1comp':       float(bic1),
        'bic_2comp':       float(bic2),
        'delta_bic':       float(delta_bic),
        'strong_evidence': bool(delta_bic > 10),
        'acute_component': {
            'mu_log':          float(mu2[acute_idx]),
            'sigma_log':       float(sig2[acute_idx]),
            'pi':              float(pi2[acute_idx]),
            'median_minutes':  float(np.exp(mu2[acute_idx])),
        },
        'chronic_component': {
            'mu_log':          float(mu2[chronic_idx]),
            'sigma_log':       float(sig2[chronic_idx]),
            'pi':              float(pi2[chronic_idx]),
            'median_minutes':  float(np.exp(mu2[chronic_idx])),
            'median_days':     float(np.exp(mu2[chronic_idx]) / 1440),
        },
        'acute_purity':    float(acute_purity),
        'chronic_purity':  float(chronic_purity),
        'all_mu':          mu2.tolist(),
        'all_sigma':       sig2.tolist(),
        'all_pi':          pi2.tolist(),
        'll_history':      ll_hist2[-20:],  # last 20 for convergence plot
    }

    out = MODEL_DIR / "em_params.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[M1] ✅ Saved → {out}")
    print(f"[M1]   Acute  median: {results['acute_component']['median_minutes']:.0f} min")
    print(f"[M1]   Chronic median: {results['chronic_component']['median_days']:.1f} days")

    # ── Figures ──────────────────────────────────────────────────────────────
    _plot_bimodal(X, mu2, sig2, pi2, FIG_DIR / "em_bimodal.png")

    return results


def _plot_bimodal(X, mu, sigma, pi, out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#0f1117')
        colors = ['#00d4ff', '#ff6b6b', '#ffd93d']

        # Left: log-space histogram + fitted components
        ax = axes[0]
        ax.set_facecolor('#1a1d2e')
        ax.hist(X, bins=80, density=True, color='#3d4166', edgecolor='none', alpha=0.8, label='Data')
        x_range = np.linspace(X.min(), X.max(), 400)
        total = np.zeros_like(x_range)
        for k in range(len(mu)):
            comp = pi[k] * norm_dist.pdf(x_range, mu[k], sigma[k])
            ax.plot(x_range, comp, color=colors[k], lw=2.5,
                    label=f'Component {k+1} (π={pi[k]:.2f}, μ={np.exp(mu[k]):.0f}min)')
            total += comp
        ax.plot(x_range, total, 'w--', lw=1.5, label='Mixture')
        ax.set_xlabel('log(resolution_mins)', color='white'); ax.set_ylabel('Density', color='white')
        ax.set_title('EM Mixture — Log-space', color='white', fontsize=13, fontweight='bold')
        ax.tick_params(colors='white'); ax.spines['bottom'].set_color('#555'); ax.spines['left'].set_color('#555')
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        ax.legend(framealpha=0.3, labelcolor='white', fontsize=9)

        # Right: original space (minutes) density
        ax2 = axes[1]
        ax2.set_facecolor('#1a1d2e')
        X_orig = np.expm1(X)
        ax2.hist(X_orig[X_orig < 1000], bins=80, density=True, color='#3d4166',
                 edgecolor='none', alpha=0.8, label='Acute events (<1000 min)')
        ax2.axvline(41, color='#00d4ff', lw=2, linestyle='--', label='Breakdown median (41 min)')
        ax2.axvline(120, color='#ff6b6b', lw=2, linestyle='--', label='Acute cluster boundary')
        ax2.set_xlabel('Resolution time (minutes)', color='white')
        ax2.set_ylabel('Density', color='white')
        ax2.set_title('Acute Cluster (< 1000 min)', color='white', fontsize=13, fontweight='bold')
        ax2.tick_params(colors='white'); ax2.spines['bottom'].set_color('#555'); ax2.spines['left'].set_color('#555')
        for sp in ['top','right']: ax2.spines[sp].set_visible(False)
        ax2.legend(framealpha=0.3, labelcolor='white', fontsize=9)

        plt.tight_layout(pad=2)
        plt.savefig(out_path, dpi=150, facecolor='#0f1117')
        plt.close()
        print(f"[M1]   Plot saved → {out_path}")
    except Exception as e:
        print(f"[M1]   Plot skipped: {e}")


if __name__ == "__main__":
    df = pd.read_parquet(DATA_DIR / "astram_clean.parquet")
    results = run_em(df)
