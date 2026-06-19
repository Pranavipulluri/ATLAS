"""
M4 — Transfer Entropy (Corridor Contagion Network)
Directed information flow between corridors.
Outputs: models/te_matrix.json, figures/te_network.png
"""

import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
from math import log2
from collections import Counter

warnings.filterwarnings('ignore')

DATA_DIR  = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"
FIG_DIR   = Path(__file__).parent.parent / "figures"
MODEL_DIR.mkdir(exist_ok=True)


def build_time_series(df: pd.DataFrame, bin_hours: float = 2.0) -> pd.DataFrame:
    """Build corridor × time-bin event count matrix."""
    named = df[
        df['corridor'].notna() &
        (df['corridor'] != 'Non-corridor') &
        df['start_datetime_IST'].notna()
    ].copy()

    # Only corridors with enough events
    corridor_counts = named['corridor'].value_counts()
    keep = corridor_counts[corridor_counts >= 150].index.tolist()
    named = named[named['corridor'].isin(keep)]

    if named.empty:
        return pd.DataFrame()

    # Create time bins
    named['time_bin'] = named['start_datetime_IST'].dt.floor(f'{int(bin_hours * 60)}min')
    ts_matrix = (
        named.groupby(['time_bin', 'corridor'])
        .size()
        .unstack(fill_value=0)
    )
    ts_matrix = ts_matrix.reindex(
        pd.date_range(ts_matrix.index.min(), ts_matrix.index.max(), freq=f'{int(bin_hours*60)}min'),
        fill_value=0
    )
    print(f"[M4]   Time series: {ts_matrix.shape[0]} bins × {ts_matrix.shape[1]} corridors")
    print(f"[M4]   Sparsity: {(ts_matrix == 0).mean().mean():.1%} zero bins")
    return ts_matrix


def discretize(series: np.ndarray, bins: int = 3) -> np.ndarray:
    """Discretize continuous series into bins (0, 1, 2)."""
    series = np.array(series, dtype=float)
    thresholds = np.percentile(series[series > 0], [50, 90]) if (series > 0).any() else [0.5, 1.0]
    result = np.zeros(len(series), dtype=int)
    result[series > thresholds[0]] = 1
    result[series > thresholds[1]] = 2
    return result


def transfer_entropy(x: np.ndarray, y: np.ndarray, lag: int = 1, bins: int = 3) -> float:
    """
    T(X→Y) = H(Y_t | Y_{t-lag}) - H(Y_t | Y_{t-lag}, X_{t-lag})
    Higher value = X carries information about Y beyond Y's own past.
    """
    xd = discretize(x, bins)
    yd = discretize(y, bins)
    n = len(xd) - lag

    if n < 20:
        return 0.0

    y_now  = yd[lag:]
    y_lag  = yd[:-lag]
    x_lag  = xd[:-lag]

    # H(Y_t | Y_{t-lag})
    joint_yy  = Counter(zip(y_now, y_lag))
    marg_ylag = Counter(y_lag)
    total = n

    h_y_given_ylag = 0.0
    for (yn, yl), c in joint_yy.items():
        p_joint = c / total
        p_cond  = marg_ylag[yl] / total
        if p_joint > 0 and p_cond > 0:
            h_y_given_ylag -= p_joint * log2(p_joint / p_cond + 1e-300)

    # H(Y_t | Y_{t-lag}, X_{t-lag})
    joint_xyy  = Counter(zip(y_now, y_lag, x_lag))
    joint_xylg = Counter(zip(y_lag, x_lag))

    h_y_given_ylag_xlg = 0.0
    for (yn, yl, xl), c in joint_xyy.items():
        p_joint = c / total
        p_cond  = joint_xylg[(yl, xl)] / total
        if p_joint > 0 and p_cond > 0:
            h_y_given_ylag_xlg -= p_joint * log2(p_joint / p_cond + 1e-300)

    te = h_y_given_ylag - h_y_given_ylag_xlg
    return max(0.0, float(te))


def shuffle_test(x: np.ndarray, y: np.ndarray, lag: int = 1,
                 bins: int = 3, n_shuffles: int = 100, seed: int = 42) -> tuple:
    """Permutation test: p-value for observed TE vs null distribution."""
    rng = np.random.default_rng(seed)
    observed = transfer_entropy(x, y, lag=lag, bins=bins)
    null_dist = []
    for _ in range(n_shuffles):
        x_shuf = rng.permutation(x)
        null_dist.append(transfer_entropy(x_shuf, y, lag=lag, bins=bins))
    null_arr = np.array(null_dist)
    p_val = float((null_arr >= observed).mean())
    null_95 = float(np.percentile(null_arr, 95))
    return observed, p_val, null_95


def run_transfer_entropy(df: pd.DataFrame) -> dict:
    print("[M4] Computing Transfer Entropy (corridor contagion)...")
    ts_matrix = build_time_series(df)
    if ts_matrix.empty:
        return {}

    corridors = ts_matrix.columns.tolist()
    n_corr = len(corridors)
    print(f"[M4]   Corridors: {corridors}")

    te_matrix   = np.zeros((n_corr, n_corr))
    p_matrix    = np.ones((n_corr, n_corr))
    significant = []

    total_pairs = n_corr * (n_corr - 1)
    done = 0
    for i, src in enumerate(corridors):
        for j, dst in enumerate(corridors):
            if i == j:
                continue
            x = ts_matrix[src].values.astype(float)
            y = ts_matrix[dst].values.astype(float)
            te, p, null95 = shuffle_test(x, y, lag=1, bins=3, n_shuffles=100)
            te_matrix[i, j] = te
            p_matrix[i, j] = p
            if p < 0.05 and te > null95:
                significant.append({
                    'src': src, 'dst': dst,
                    'te': round(te, 5), 'p': round(p, 4),
                })
            done += 1
            if done % 10 == 0:
                print(f"[M4]   Progress: {done}/{total_pairs} pairs", end='\r')

    print(f"\n[M4]   Significant contagion pathways: {len(significant)}")

    # Contagion network centrality
    import networkx as nx
    G = nx.DiGraph()
    for edge in significant:
        G.add_edge(edge['src'], edge['dst'], weight=edge['te'])

    out_strength = {n: round(d, 4) for n, d in G.out_degree(weight='weight')}
    in_strength  = {n: round(d, 4) for n, d in G.in_degree(weight='weight')}
    print("\n[M4]   Top source corridors (high out-strength -> contagion sources):")
    for c, s in sorted(out_strength.items(), key=lambda x: -x[1])[:5]:
        print(f"    {c}: {s:.4f}")

    # Save
    results = {
        'corridors': corridors,
        'te_matrix': te_matrix.tolist(),
        'p_matrix': p_matrix.tolist(),
        'significant_edges': sorted(significant, key=lambda x: -x['te']),
        'out_strength': out_strength,
        'in_strength': in_strength,
    }
    out = MODEL_DIR / "te_matrix.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[M4] ✅ Saved → {out}")

    _plot_te_heatmap(corridors, te_matrix, p_matrix, FIG_DIR / "te_heatmap.png")
    _plot_te_network(G, out_strength, FIG_DIR / "te_network.png")

    return results


def _plot_te_heatmap(corridors, te_matrix, p_matrix, out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n = len(corridors)
        sig_mask = p_matrix < 0.05
        display = np.where(sig_mask, te_matrix, 0)
        labels = [c.replace(' Road', ' Rd').replace('Bellary', 'Bell.').replace('Bannerghata', 'Banner.') for c in corridors]

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#1a1d2e')
        im = ax.imshow(display, cmap='plasma', aspect='auto')
        plt.colorbar(im, ax=ax, label='Transfer Entropy (significant only)')
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8, color='white')
        ax.set_yticklabels(labels, fontsize=8, color='white')
        ax.set_title('Transfer Entropy Matrix\n(row→col = contagion direction, p<0.05 only)',
                     color='white', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, facecolor='#0f1117', bbox_inches='tight')
        plt.close()
        print(f"[M4]   Plot saved → {out_path}")
    except Exception as e:
        print(f"[M4]   Plot skipped: {e}")


def _plot_te_network(G, out_strength, out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import networkx as nx

        if G.number_of_edges() == 0:
            print("[M4]   No significant edges — skipping network plot")
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#0f1117')

        pos = nx.spring_layout(G, k=2.0, seed=42)
        max_out = max(out_strength.values()) if out_strength else 1
        node_sizes  = [500 + 3000 * (out_strength.get(n, 0) / (max_out + 1e-6)) for n in G.nodes()]
        node_colors = ['#ff6b6b' if out_strength.get(n, 0) > 0.02 else '#00d4ff' for n in G.nodes()]

        weights = [G[u][v]['weight'] for u, v in G.edges()]
        mx = max(weights) if weights else 1
        edge_widths = [0.5 + 4 * w / mx for w in weights]

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, font_color='white', ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='#8888ff', alpha=0.7,
                               arrows=True, arrowsize=15, ax=ax,
                               connectionstyle='arc3,rad=0.15')

        ax.set_title('Corridor Contagion Network (Transfer Entropy)\nRed = contagion SOURCE corridors',
                     color='white', fontsize=12, fontweight='bold', pad=15)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, facecolor='#0f1117', bbox_inches='tight')
        plt.close()
        print(f"[M4]   Plot saved → {out_path}")
    except Exception as e:
        print(f"[M4]   Plot skipped: {e}")


if __name__ == "__main__":
    df = pd.read_parquet(DATA_DIR / "astram_clean.parquet")
    run_transfer_entropy(df)
