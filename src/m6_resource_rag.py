"""
M6 — ResourceRAG (Dispatch Recommender)
Vector similarity search over historical closed events.
Outputs: models/rag_index.npz, models/rag_meta.parquet
"""

import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')

DATA_DIR  = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"
FIG_DIR   = Path(__file__).parent.parent / "figures"
MODEL_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

CAUSE_ORDER = [
    'vehicle_breakdown', 'accident', 'congestion', 'pot_holes', 'road_conditions',
    'water_logging', 'construction', 'tree_fall', 'public_event', 'procession',
    'vip_movement', 'protest', 'others', 'unknown',
]
CORRIDOR_ORDER = [
    'Mysore Road', 'Bellary Road 1', 'Tumkur Road', 'Bellary Road 2', 'Hosur Road',
    'ORR North 1', 'Old Madras Road', 'Magadi Road', 'ORR East 1', 'ORR North 2',
    'West of Chord Road', 'ORR West 1', 'Bannerghatta Road', 'Non-corridor',
]
VEH_ORDER = [
    'bmtc_bus', 'heavy_vehicle', 'lcv', 'private_bus', 'private_car', 'truck',
    'ksrtc_bus', 'auto', 'two_wheeler', 'others', 'N/A', 'unknown',
]
CLASS_ORDER = ['acute', 'chronic', 'planned', 'unknown']

_encoders = {}
_scaler   = None
_nn       = None
_meta_df  = None


def _safe_encode(val, order, default=0):
    try:
        return order.index(str(val))
    except ValueError:
        return default


def build_feature_vector(row: pd.Series) -> np.ndarray:
    """Build 12-dimensional feature vector for a single event."""
    cause    = _safe_encode(row.get('event_cause_clean', ''),   CAUSE_ORDER)
    corridor = _safe_encode(row.get('corridor', ''),             CORRIDOR_ORDER)
    priority = 1.0 if str(row.get('priority', '')) == 'High' else 0.0
    closure  = 1.0 if str(row.get('requires_road_closure', '')) == 'TRUE' else 0.0
    veh      = _safe_encode(row.get('veh_type_imputed', ''),     VEH_ORDER)
    cls      = _safe_encode(row.get('event_class', ''),          CLASS_ORDER)
    hour     = float(row.get('hour_IST', 12)) / 24.0
    dow      = float(row.get('day_of_week', 3)) / 7.0
    month    = float(row.get('month', 1)) / 12.0
    night    = 1.0 if row.get('is_night', False) else 0.0
    etype    = 1.0 if str(row.get('event_type', '')) == 'planned' else 0.0
    chronic  = 1.0 if str(row.get('event_class', '')) == 'chronic' else 0.0

    return np.array([cause, corridor, priority, closure, veh, cls,
                     hour, dow, month, night, etype, chronic], dtype=float)


def _compute_efficiency_weight(resolution_mins: float, baseline_mins: float) -> float:
    """
    Intervention Efficiency = baseline / actual
    If event resolved faster than baseline → ratio > 1 → good intervention.
    If slower → ratio < 1 → poor/failed intervention.

    Weight tiers (for ResourceRAG retrieval):
        Excellent  (ratio > 1.2) → 1.0
        Good       (0.9–1.2)    → 0.8
        Poor       (0.7–0.9)    → 0.4
        Failed     (< 0.7)      → 0.1
    """
    if baseline_mins <= 0 or resolution_mins <= 0:
        return 0.5  # unknown → neutral
    ratio = baseline_mins / resolution_mins
    if ratio > 1.2:
        return 1.0   # Excellent
    elif ratio >= 0.9:
        return 0.8   # Good
    elif ratio >= 0.7:
        return 0.4   # Poor
    else:
        return 0.1   # Failed


def build_index(df: pd.DataFrame) -> dict:
    global _scaler, _nn, _meta_df

    print("[M6] Building ResourceRAG index...")
    df_valid = df[df['resolution_valid']].dropna(subset=['resolution_mins']).copy()
    print(f"[M6]   N (closed events with resolution): {len(df_valid)}")

    # ── Feature matrix ───────────────────────────────────────────────────────
    features = np.vstack([build_feature_vector(row) for _, row in df_valid.iterrows()])
    _scaler = StandardScaler()
    features_scaled = _scaler.fit_transform(features)

    # ── Resolution quality score (relative rank within cause group) ──────────
    df_valid['resolution_score'] = df_valid.groupby('event_cause_clean')['resolution_mins'].transform(
        lambda x: 1 - (x.rank(method='average') / len(x))
    )

    # ── Intervention Efficiency Score ─────────────────────────────────────────
    # Baseline: median resolution per cause (the M1 expected duration)
    cause_baselines = df_valid.groupby('event_cause_clean')['resolution_mins'].median()
    df_valid['baseline_mins'] = df_valid['event_cause_clean'].map(cause_baselines)
    df_valid['success_weight'] = df_valid.apply(
        lambda r: _compute_efficiency_weight(r['resolution_mins'], r['baseline_mins']), axis=1
    )

    # Log the breakdown
    w_counts = df_valid['success_weight'].value_counts().sort_index(ascending=False)
    excellent = (df_valid['success_weight'] == 1.0).sum()
    failed    = (df_valid['success_weight'] == 0.1).sum()
    print(f"[M6]   Efficiency scoring: Excellent={excellent} ({100*excellent/len(df_valid):.0f}%)  "
          f"Failed={failed} ({100*failed/len(df_valid):.0f}%)")

    # ── Combined retrieval score: quality × success_weight × priority ────────
    df_valid['adjusted_score'] = (
        df_valid['resolution_score']
        * df_valid['success_weight']
        * df_valid['priority'].map({'High': 2.0, 'Low': 1.0}).fillna(1.0)
    )

    # ── Nearest neighbours index ─────────────────────────────────────────────
    _nn = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
    _nn.fit(features_scaled)

    # ── Save ─────────────────────────────────────────────────────────────────
    _meta_df = df_valid.reset_index(drop=True)
    np.savez_compressed(MODEL_DIR / "rag_features.npz", features=features_scaled)
    _meta_df[[
        'id', 'event_cause_clean', 'event_class', 'corridor', 'zone_imputed',
        'priority', 'requires_road_closure', 'veh_type_imputed',
        'hour_IST', 'day_of_week', 'month', 'is_night', 'event_type',
        'resolution_mins', 'baseline_mins', 'resolution_score',
        'success_weight', 'adjusted_score',
    ]].to_parquet(MODEL_DIR / "rag_meta.parquet", index=False)

    scaler_params = {
        'mean': _scaler.mean_.tolist(),
        'scale': _scaler.scale_.tolist(),
        'feature_names': ['cause','corridor','priority','closure','veh','class',
                          'hour','dow','month','night','planned','chronic'],
    }
    with open(MODEL_DIR / "rag_scaler.json", 'w') as f:
        json.dump(scaler_params, f, indent=2)

    print(f"[M6] ✅ Index built — {len(df_valid)} events indexed")
    print(f"[M6]   Feature dims: {features_scaled.shape[1]}")
    return {
        'n_indexed': len(df_valid),
        'feature_dims': features_scaled.shape[1],
        'pct_excellent': round(100 * excellent / len(df_valid), 1),
        'pct_failed': round(100 * failed / len(df_valid), 1),
    }


def load_index():
    """Load pre-built index from disk."""
    global _scaler, _nn, _meta_df
    data = np.load(MODEL_DIR / "rag_features.npz")
    features = data['features']
    _meta_df = pd.read_parquet(MODEL_DIR / "rag_meta.parquet")

    with open(MODEL_DIR / "rag_scaler.json") as f:
        sp = json.load(f)
    _scaler = StandardScaler()
    _scaler.mean_  = np.array(sp['mean'])
    _scaler.scale_ = np.array(sp['scale'])
    _scaler.n_features_in_ = len(sp['mean'])

    _nn = NearestNeighbors(n_neighbors=min(10, len(features)), metric='cosine', algorithm='brute')
    _nn.fit(features)


def query(event_dict: dict, k: int = 5) -> dict:
    """
    Query ResourceRAG with a new event.
    event_dict keys: event_cause_clean, corridor, priority, requires_road_closure,
                     veh_type_imputed, event_class, hour_IST, day_of_week, month,
                     is_night, event_type
    """
    global _scaler, _nn, _meta_df
    if _nn is None:
        load_index()

    row = pd.Series(event_dict)
    fv  = build_feature_vector(row).reshape(1, -1)
    fv_scaled = _scaler.transform(fv)

    # Find k nearest neighbours
    distances, indices = _nn.kneighbors(fv_scaled, n_neighbors=min(k + 3, len(_meta_df)))
    neighbours = _meta_df.iloc[indices[0]].copy()
    neighbours['similarity'] = 1 - distances[0]  # cosine: 1-dist = similarity

    # ── Success-weighted combined score ──────────────────────────────────────
    # similarity × success_weight → filters out historically bad interventions
    sw = neighbours.get('success_weight', pd.Series([0.8] * len(neighbours)))
    neighbours['combined'] = neighbours['similarity'] * sw * neighbours.get('adjusted_score',
                                                                             pd.Series([1.0] * len(neighbours)))
    neighbours = neighbours.nlargest(k, 'combined')

    # Fallback: if all similarities are low (<0.4), use cause-only lookup
    if neighbours['similarity'].max() < 0.4:
        cause_matches = _meta_df[_meta_df['event_cause_clean'] == event_dict.get('event_cause_clean', '')]
        if len(cause_matches) >= 3:
            neighbours = cause_matches.nlargest(k, 'adjusted_score')
            neighbours['similarity'] = 0.3
            neighbours['fallback'] = True

    # Weighted statistics
    weights = neighbours['combined'].values
    w_sum   = weights.sum() + 1e-9

    exp_res = float(np.average(neighbours['resolution_mins'], weights=weights))
    q25     = float(neighbours['resolution_mins'].quantile(0.25))
    q75     = float(neighbours['resolution_mins'].quantile(0.75))
    conf    = float(neighbours['similarity'].mean())

    # Road closure consensus
    closure_needed = bool((neighbours['requires_road_closure'] == 'TRUE').any())

    # Format human-readable resolution time
    if exp_res < 120:
        res_str = f"{exp_res:.0f} minutes"
    elif exp_res < 1440:
        res_str = f"{exp_res/60:.1f} hours"
    else:
        res_str = f"{exp_res/1440:.1f} days"

    avg_success = float(neighbours.get('success_weight', pd.Series([0.8]*len(neighbours))).mean())

    card = {
        'recommended_priority':     neighbours['priority'].mode().iloc[0] if len(neighbours) else 'High',
        'road_closure_recommended': closure_needed,
        'expected_resolution':      res_str,
        'expected_resolution_mins': round(exp_res, 1),
        'resolution_iqr_mins':      [round(q25, 1), round(q75, 1)],
        'confidence_score':         round(conf, 3),
        'confidence_label':         'HIGH' if conf > 0.7 else 'MEDIUM' if conf > 0.4 else 'LOW',
        'avg_intervention_success': round(avg_success, 3),
        'intervention_quality':     'EXCELLENT' if avg_success >= 0.9 else 'GOOD' if avg_success >= 0.7 else 'POOR',
        'n_similar_events':         len(neighbours),
        'similar_event_ids':        neighbours['id'].tolist(),
        'similar_event_causes':     neighbours['event_cause_clean'].tolist(),
        'similar_event_corridors':  neighbours['corridor'].tolist(),
        'low_confidence_warning':   conf < 0.4,
        'fallback_used':            bool(neighbours.get('fallback', pd.Series([False])).any()),
    }

    if conf < 0.4:
        card['warning'] = (
            "LOW CONFIDENCE: No closely similar historical events found. "
            "Recommendation is based on same-cause events only. "
            "Apply domain expertise."
        )

    return card


def run_resource_rag(df: pd.DataFrame) -> dict:
    result = build_index(df)
    _plot_resolution_distributions(df, FIG_DIR / "rag_distributions.png")
    return result


def _plot_resolution_distributions(df, out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        df_res = df[df['resolution_valid']].dropna(subset=['resolution_mins'])
        top_causes = df_res['event_cause_clean'].value_counts().head(8).index.tolist()
        palette = ['#00d4ff','#ff6b6b','#ffd93d','#a855f7','#10b981','#f97316','#ec4899','#8b5cf6']

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#1a1d2e')

        for cause, col in zip(top_causes, palette):
            data = df_res[df_res['event_cause_clean'] == cause]['log_resolution_mins'].dropna()
            if len(data) < 5:
                continue
            import numpy as np
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data, bw_method='scott')
            x = np.linspace(data.min(), data.max(), 200)
            ax.plot(x, kde(x), color=col, lw=2.5, label=f'{cause} (N={len(data)})')
            ax.fill_between(x, kde(x), alpha=0.1, color=col)

        ax.axvline(np.log1p(120), color='white', lw=1, linestyle=':', alpha=0.7,
                   label='Acute/Chronic boundary (~2h)')
        ax.set_xlabel('log(resolution_mins)', color='white')
        ax.set_ylabel('Density', color='white')
        ax.set_title('Resolution Time Distributions by Cause\n(ResourceRAG training data)',
                     color='white', fontsize=12, fontweight='bold')
        ax.tick_params(colors='white')
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        for sp in ['bottom','left']: ax.spines[sp].set_color('#555')
        ax.legend(framealpha=0.3, labelcolor='white', fontsize=9, ncol=2)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, facecolor='#0f1117')
        plt.close()
        print(f"[M6]   Plot saved → {out_path}")
    except Exception as e:
        print(f"[M6]   Plot skipped: {e}")


if __name__ == "__main__":
    df = pd.read_parquet(DATA_DIR / "astram_clean.parquet")
    run_resource_rag(df)

    # Test query
    test_event = {
        'event_cause_clean': 'vehicle_breakdown',
        'corridor': 'Mysore Road',
        'priority': 'High',
        'requires_road_closure': 'FALSE',
        'veh_type_imputed': 'bmtc_bus',
        'event_class': 'acute',
        'hour_IST': 22,
        'day_of_week': 1,
        'month': 12,
        'is_night': True,
        'event_type': 'unplanned',
    }
    card = query(test_event)
    print("\n[M6] Test query result:")
    for k, v in card.items():
        print(f"  {k}: {v}")
