"""
M0 — Data Pipeline
Cleans and prepares the Astram event dataset for all downstream models.
Outputs: data/astram_clean.parquet
"""

import pandas as pd
import numpy as np
import re
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

RAW_CSV = Path(__file__).parent.parent / "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"

ACUTE_CAUSES   = {'vehicle_breakdown', 'accident', 'congestion', 'procession', 'protest'}
CHRONIC_CAUSES = {'pot_holes', 'road_conditions', 'water_logging', 'construction', 'tree_fall'}

# Police station → zone lookup (built from dataset knowledge)
STATION_ZONE = {
    'Peenya': 'West Zone 1', 'Yeshwantpur': 'West Zone 1', 'Rajajinagar': 'West Zone 1',
    'Mathikere': 'West Zone 1', 'Jalahalli': 'West Zone 1', 'Nagasandra': 'West Zone 1',
    'Hesaraghatta': 'North Zone 1', 'Yelahanka': 'North Zone 1', 'Sahakar Nagar': 'North Zone 1',
    'Devanahalli': 'North Zone 2', 'Jakkur': 'North Zone 2', 'Bagalur': 'North Zone 2',
    'Hebbal': 'North Zone 2', 'RT Nagar': 'North Zone 2', 'Sadashivanagar': 'Central Zone 1',
    'Basaveshwaranagar': 'West Zone 2', 'Vijayanagar': 'West Zone 2', 'Kengeri': 'West Zone 2',
    'Uttarahalli': 'West Zone 2', 'Rajarajeshwari Nagar': 'West Zone 2',
    'Tilak Nagar': 'West Zone 2', 'Magadi Road': 'West Zone 2',
    'Shivajinagar': 'Central Zone 1', 'Cubbon Park': 'Central Zone 1',
    'Ulsoor': 'Central Zone 2', 'Halasuru': 'Central Zone 2', 'Frazer Town': 'Central Zone 2',
    'Banaswadi': 'Central Zone 2', 'KR Puram': 'East Zone 1', 'Kadugodi': 'East Zone 1',
    'Whitefield': 'East Zone 1', 'Varthur': 'East Zone 2', 'Sarjapur': 'East Zone 2',
    'Bellandur': 'East Zone 2', 'HSR Layout': 'South Zone 2', 'Koramangala': 'South Zone 2',
    'BTM Layout': 'South Zone 2', 'Madiwala': 'South Zone 2', 'Bommanahalli': 'South Zone 2',
    'Electronic City': 'South Zone 1', 'Begur': 'South Zone 1', 'Hulimavu': 'South Zone 1',
    'Bannerghatta Road': 'South Zone 1', 'JP Nagar': 'South Zone 1',
    'Jayanagar': 'South Zone 1', 'Basavanagudi': 'South Zone 1', 'Wilson Garden': 'South Zone 1',
    'Anand Nagar': 'West Zone 2', 'Nandini Layout': 'West Zone 2',
}

KEYWORD_REMAP = {
    'water': 'water_logging', 'flood': 'water_logging', 'rain': 'water_logging', 'waterlog': 'water_logging',
    'tree': 'tree_fall', 'branch': 'tree_fall', 'fallen tree': 'tree_fall',
    'accident': 'accident', 'collision': 'accident', 'crash': 'accident', 'hit': 'accident',
    'debris': 'tree_fall', 'garbage': 'tree_fall',
    'pothole': 'pot_holes', 'pot hole': 'pot_holes', 'pit': 'pot_holes',
    'construction': 'construction', 'repair': 'construction', 'work': 'construction',
    'breakdown': 'vehicle_breakdown', 'broke': 'vehicle_breakdown', 'stalled': 'vehicle_breakdown',
    'congestion': 'congestion', 'traffic jam': 'congestion', 'jam': 'congestion',
}

def load_and_clean() -> pd.DataFrame:
    print("[M0] Loading raw CSV...")
    df = pd.read_csv(RAW_CSV, encoding='utf-8', encoding_errors='replace', low_memory=False)
    print(f"[M0] Loaded {len(df)} rows, {len(df.columns)} columns")

    # ── 1. Timezone-correct timestamps ──────────────────────────────────────
    print("[M0] Fixing timezones (UTC → IST)...")
    for col in ['start_datetime', 'closed_datetime', 'end_datetime']:
        df[col] = pd.to_datetime(df[col], format='mixed', utc=True, errors='coerce')
        df[f'{col}_IST'] = df[col].dt.tz_convert('Asia/Kolkata')

    # Flag rows that had to fall back on created_date
    df['timestamp_source'] = 'start_datetime'
    bad_start = df['start_datetime_IST'].isna()
    df.loc[bad_start, 'timestamp_source'] = 'created_date'
    if bad_start.sum() > 0:
        df['created_date'] = pd.to_datetime(df['created_date'], format='mixed', utc=True, errors='coerce')
        df.loc[bad_start, 'start_datetime_IST'] = df.loc[bad_start, 'created_date'].dt.tz_convert('Asia/Kolkata')

    df['hour_IST']    = df['start_datetime_IST'].dt.hour
    df['day_of_week'] = df['start_datetime_IST'].dt.dayofweek
    df['month']       = df['start_datetime_IST'].dt.month
    df['is_night']    = df['hour_IST'].apply(lambda h: h >= 20 or h < 6)

    # ── 2. Resolution time ───────────────────────────────────────────────────
    print("[M0] Computing resolution times...")
    df['resolution_mins'] = (
        (df['closed_datetime_IST'] - df['start_datetime_IST'])
        .dt.total_seconds() / 60
    )
    valid_mask = (df['resolution_mins'] > 0) & (df['resolution_mins'] < 200_000)
    df['resolution_valid'] = valid_mask
    df.loc[~valid_mask, 'resolution_mins'] = np.nan
    df['log_resolution_mins'] = np.log1p(df['resolution_mins'])
    print(f"[M0]   Valid resolution times: {valid_mask.sum()} / {len(df)}")

    # ── 3. Cause reclassification ────────────────────────────────────────────
    print("[M0] Reclassifying 'others' cause using description keywords...")
    df['event_cause'] = df['event_cause'].fillna('others').str.strip()
    df['event_cause_clean'] = df['event_cause'].copy()

    others_mask = df['event_cause'] == 'others'
    def reclassify(desc):
        if not isinstance(desc, str):
            return 'others'
        d = desc.lower()
        for kw, cause in KEYWORD_REMAP.items():
            if kw in d:
                return cause
        return 'others'
    df.loc[others_mask, 'event_cause_clean'] = df.loc[others_mask, 'description'].apply(reclassify)

    # Merge Debris → tree_fall; drop test_demo
    df['event_cause_clean'] = df['event_cause_clean'].replace('Debris', 'tree_fall')
    df = df[df['event_cause_clean'] != 'test_demo'].copy()
    print(f"[M0]   Cause distribution:\n{df['event_cause_clean'].value_counts().head(10).to_string()}")

    # ── 4. Event class label ─────────────────────────────────────────────────
    def assign_class(row):
        if row['event_type'] == 'planned':
            return 'planned'
        c = row['event_cause_clean']
        if c in ACUTE_CAUSES:   return 'acute'
        if c in CHRONIC_CAUSES: return 'chronic'
        return 'unknown'
    df['event_class'] = df.apply(assign_class, axis=1)

    # ── 5. Zone imputation ───────────────────────────────────────────────────
    print("[M0] Imputing missing zones from police_station lookup...")
    df['zone'] = df['zone'].replace('NULL', np.nan).replace('', np.nan)
    df['zone_imputed'] = df['zone'].copy()
    zone_missing = df['zone_imputed'].isna()
    df.loc[zone_missing, 'zone_imputed'] = (
        df.loc[zone_missing, 'police_station']
        .map(STATION_ZONE)
    )
    df['zone_imputed'] = df['zone_imputed'].fillna('zone_unknown')
    print(f"[M0]   Zone fill rate: {(~df['zone_imputed'].isin(['zone_unknown', ''])).mean():.1%}")

    # ── 6. veh_type imputation (NLP for breakdowns) ──────────────────────────
    print("[M0] Imputing veh_type for vehicle_breakdown events...")
    df['veh_type'] = df['veh_type'].replace('NULL', '').fillna('')
    df['veh_type_imputed'] = df['veh_type'].copy()

    # Non-breakdown: N/A
    df.loc[df['event_cause_clean'] != 'vehicle_breakdown', 'veh_type_imputed'] = 'N/A'

    # For breakdown events with missing veh_type — use keyword rules on description
    bd_missing = (df['event_cause_clean'] == 'vehicle_breakdown') & (df['veh_type'] == '')
    VEH_KEYWORDS = {
        'bmtc': 'bmtc_bus', 'bus': 'bmtc_bus', 'ksrtc': 'ksrtc_bus',
        'truck': 'truck', 'lorry': 'truck', 'tanker': 'truck',
        'heavy': 'heavy_vehicle', 'hv': 'heavy_vehicle',
        'lcv': 'lcv', 'auto': 'auto', 'bike': 'two_wheeler',
        'car': 'private_car', 'cab': 'private_car', 'suv': 'private_car',
    }
    def impute_veh(desc):
        if not isinstance(desc, str): return 'unknown'
        d = desc.lower()
        for kw, vt in VEH_KEYWORDS.items():
            if kw in d: return vt
        return 'unknown'
    df.loc[bd_missing, 'veh_type_imputed'] = df.loc[bd_missing, 'description'].apply(impute_veh)
    df['veh_type_imputed'] = df['veh_type_imputed'].replace('', 'unknown')

    # ── 7. Lat/lon validation ────────────────────────────────────────────────
    df['latitude']  = pd.to_numeric(df['latitude'],  errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    # Bengaluru bounds: lat 12.7–13.2, lon 77.4–77.8
    valid_coords = (
        df['latitude'].between(12.7, 13.2) &
        df['longitude'].between(77.4, 77.8)
    )
    df.loc[~valid_coords, ['latitude', 'longitude']] = np.nan

    # ── 8. Corridor cleanup ──────────────────────────────────────────────────
    df['corridor'] = df['corridor'].fillna('Non-corridor').replace('', 'Non-corridor')

    # ── 9. Infrastructure group for EVT ─────────────────────────────────────
    df['infrastructure_group'] = df['event_cause_clean'].map({
        'pot_holes': 'road_surface',
        'road_conditions': 'road_surface',
        'water_logging': 'drainage',
        'construction': 'construction',
        'tree_fall': 'vegetation',
    }).fillna('N/A')

    # ── 10. Select and save ──────────────────────────────────────────────────
    keep = [
        'id', 'event_cause_clean', 'event_class', 'event_type',
        'latitude', 'longitude', 'corridor', 'zone_imputed', 'police_station',
        'priority', 'requires_road_closure', 'veh_type_imputed',
        'start_datetime_IST', 'closed_datetime_IST',
        'resolution_mins', 'log_resolution_mins', 'resolution_valid',
        'hour_IST', 'day_of_week', 'month', 'is_night',
        'infrastructure_group', 'timestamp_source',
        'address', 'description', 'junction',
    ]
    keep_existing = [c for c in keep if c in df.columns]
    df_out = df[keep_existing].copy()
    out_path = DATA_DIR / "astram_clean.parquet"
    df_out.to_parquet(out_path, index=False)
    print(f"[M0] ✅ Saved {len(df_out)} rows → {out_path}")

    # Summary stats
    summary = {
        'total_rows': len(df_out),
        'valid_resolution': int(df_out['resolution_valid'].sum()),
        'class_counts': df_out['event_class'].value_counts().to_dict(),
        'cause_counts': df_out['event_cause_clean'].value_counts().to_dict(),
        'corridor_counts': df_out['corridor'].value_counts().head(15).to_dict(),
    }
    with open(DATA_DIR / "pipeline_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    return df_out


if __name__ == "__main__":
    df = load_and_clean()
    print("\n[M0] Final shape:", df.shape)
    print("[M0] Event classes:\n", df['event_class'].value_counts())
    print("[M0] Cause distribution:\n", df['event_cause_clean'].value_counts())
