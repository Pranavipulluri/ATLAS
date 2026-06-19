import sys; sys.path.insert(0, '.')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import pandas as pd

df = pd.read_parquet('data/astram_clean.parquet')

# Rebuild M6 with efficiency scores
print('=== Rebuilding M6 with efficiency scores ===')
from src.m6_resource_rag import run_resource_rag
r = run_resource_rag(df)
print('  Excellent: {}%  Failed: {}%'.format(r['pct_excellent'], r['pct_failed']))
print()

# Run M8
print('=== Running M8 Event Impact Simulator ===')
from src.m8_event_simulator import run_event_simulator
run_event_simulator(df)
print()

# NLP smoke test
print('=== NLP Parser smoke test ===')
from bot.nlp_parser import parse_incident
tests = [
    'Heavy rainfall and a truck breakdown on Mysore Road near Kengeri',
    'Accident at Silk Board junction, multiple vehicles involved',
    'IPL match at Chinnaswamy Stadium tomorrow evening, expecting 50000 crowd',
    'VIP convoy from Vidhana Soudha at 10am Monday',
    'Tree fell on Tumkur Road blocking one lane',
]
for t in tests:
    p = parse_incident(t)
    mode = p['mode']
    ex   = p['extracted']
    conf = p['confidence']
    if mode == 'dispatch':
        print('  [{}] {:20s} | {:20s} | conf={:.2f}'.format(
            mode, str(ex.get('cause','')), str(ex.get('corridor','')), conf))
    else:
        print('  [{}] {:20s} | {:20s} | crowd={}'.format(
            mode, str(ex.get('type','')), str(ex.get('venue','')), ex.get('crowd')))
