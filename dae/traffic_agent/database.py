import sqlite3
import json
import os

# Create DB file in backend folder
DB_PATH = os.path.join(os.path.dirname(__file__), "emergency.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS emergency_routes (
            vehicle_id TEXT PRIMARY KEY,
            route_plan TEXT,
            current_step INTEGER,
            status TEXT
        )
    ''')
    conn.commit()
    conn.close()

def register_emergency(vehicle_id: str, route_plan: list):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO emergency_routes (vehicle_id, route_plan, current_step, status)
        VALUES (?, ?, ?, ?)
    ''', (vehicle_id, json.dumps(route_plan), 0, 'ACTIVE'))
    conn.commit()
    conn.close()

def update_emergency_step(vehicle_id: str, step: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        UPDATE emergency_routes SET current_step = ? WHERE vehicle_id = ?
    ''', (step, vehicle_id))
    conn.commit()
    conn.close()

def complete_emergency(vehicle_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        UPDATE emergency_routes SET status = 'COMPLETED' WHERE vehicle_id = ?
    ''', (vehicle_id,))
    conn.commit()
    conn.close()

def get_approaching_emergency(intersection_id: str):
    """
    Returns the entry_lane if an emergency vehicle is currently approaching this intersection.
    If multiple, returns the first one found.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT route_plan, current_step FROM emergency_routes WHERE status = 'ACTIVE'
    ''')
    active_routes = c.fetchall()
    conn.close()
    
    for route_json, current_step in active_routes:
        route_plan = json.loads(route_json)
        # Check if the ambulance is still on the route
        if current_step < len(route_plan):
            current_target = route_plan[current_step]
            if current_target.get("intersection_id") == intersection_id:
                return current_target.get("entry_lane")
    return None
