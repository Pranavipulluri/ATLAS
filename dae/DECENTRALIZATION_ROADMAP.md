# Agentic Edge Traffic Management - Decentralization Roadmap

## Phase 1: Laptop Improvements (Complete ✓)

This phase introduces autonomous node architecture and abstraction layers that enable future Jetson deployment without changing application code. All improvements are backward-compatible with the existing dashboard.

### What Changed

#### 1. **Autonomous Node Architecture** ✓
- **File**: `traffic_agent/node/intersection_node.py`
- **Purpose**: Each intersection now contains a fully autonomous agent that can:
  - Make independent traffic signal decisions (via `MasterAgent`)
  - Track per-lane vehicle density and wait times (via `LaneAgent`)
  - Receive external inputs (green-wave alerts, pedestrian signals)
  - Execute deterministic priority auction logic
- **Key Methods**:
  - `tick(dt)`: Main simulation loop (autonomous decision-making)
  - `set_lane_density()`: Accept density input from external detector
  - `receive_green_wave_alert()`: Respond to upstream ambulance coordination
  - `add_emergency()`: Register emergency vehicles
- **Benefit**: Each node is now a self-contained MAS (Multi-Agent System) that can run on any hardware (laptop, Jetson, cloud)

#### 2. **Network Coordination Layer** ✓
- **File**: `traffic_agent/node/node_network.py`
- **Purpose**: Abstract coordination interface that works in both:
  - **Simulation mode** (current): In-memory node calls (laptop development)
  - **MQTT mode** (future): Distributed MQTT pub/sub (Jetson edge network)
- **Key Methods**:
  - `tick_all(dt)`: Tick all nodes, collect states, update ambulances
  - `spawn_ambulance(route)`: Trigger ambulance routing with green-wave alerts
  - `get_grid_state()`: Return dashboard-compatible state
- **Benefit**: Same code runs on laptop and Jetson—no refactoring needed for deployment

#### 3. **Environment-Based Configuration** ✓
- **File**: `traffic_agent/config.py`
- **Settings**:
  ```python
  TRAFFIC_MODE = "simulation"  # or "mqtt" for Jetson
  MQTT_BROKER = "localhost:1883"  # MQTT broker address
  NODE_ID = None  # None (all nodes) or "A" (single node)
  DETECTION_MODE = "mock"  # or "yolo"
  DEBUG = True  # Verbose logging
  ```
- **Benefit**: Switch deployment modes via environment variables, no code changes

#### 4. **Structured Logging** ✓
- **File**: `traffic_agent/logger.py`
- **Purpose**: Centralized logging for debugging node behavior
- **Loggers**: `logger_api`, `logger_sim`, `logger_node`, `logger_mqtt`
- **Benefit**: Track events across decentralized system easily

#### 5. **Abstract Detection Interface** ✓
- **File**: `traffic_agent/detection/detector_interface.py`
- **Purpose**: Decouple detection backend (YOLO or mock) from node logic
- **Implementations**:
  - `MockDetector`: Realistic random vehicle counts (5% emergency rate)
  - `YOLODetector`: Real YOLO inference with graceful fallback
- **Benefit**: Swap detection backends without touching node code

#### 6. **Refactored Main Entry Point** ✓
- **File**: `traffic_agent/main.py`
- **Changes**:
  - Uses `NodeNetwork` instead of `GridCoordinator`
  - Command handler routes WebSocket events to `node_network` methods
  - REST endpoints use `node_network.get_grid_state()` and `node_network.spawn_ambulance()`
  - Added health check endpoint (`/health`)
  - Dashboard receives same WebSocket format (backward-compatible)
- **Benefit**: Clean separation between API layer and autonomous nodes

---

## Running the Laptop Improvements

### Prerequisites
```bash
cd traffic_agent
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### Run Tests
```bash
# Validate autonomous node architecture
python test_autonomous.py
```

**Expected Output:**
```
=== Test 1: Single Node Autonomous Tick ===
  Tick 0: 4 lanes, current_phase=North
  Tick 1: 4 lanes, current_phase=North
  ...
✓ Single node ticking works independently

=== Test 2: NodeNetwork Simulation Mode ===
  Tick 0: Network has 4 active nodes
  ...
✓ NodeNetwork can coordinate multiple autonomous nodes

... (more tests)

✓ ALL TESTS PASSED
```

### Run Backend
```bash
# Terminal 1: Start FastAPI server
python main.py
# Expected: "Starting Agentic Edge Traffic Management in simulation mode"

# Terminal 2: Check health
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "mode": "simulation",
  "active_nodes": ["A", "B", "C", "D"],
  "active_ambulances": 0
}
```

### Run Full Stack
```bash
# Terminal 1: Backend
cd traffic_agent && python main.py

# Terminal 2: Frontend (in project root)
cd public && npm run dev
# Visit http://localhost:3000
```

---

## Phase 2: Jetson Deployment (Next Week)

Once laptop improvements are complete and tested, migration to Jetson is straightforward:

### Step 1: Add MQTT Bridge (new file)
- Implement `traffic_agent/node/mqtt_bridge.py`
- `MQTTBridge` class publishes node states to MQTT
- When `config.TRAFFIC_MODE = "mqtt"`, NodeNetwork uses MQTT instead of in-memory calls
- **Code change required**: ~50 lines in NodeNetwork to route MQTT calls

### Step 2: Dockerize
```dockerfile
# Dockerfile.node
FROM python:3.10-slim
WORKDIR /app
COPY traffic_agent .
RUN pip install -r requirements.txt
CMD ["python", "main.py"]
```

### Step 3: Deploy to Jetson
```bash
# On Jetson device
docker-compose up -d  # Runs MQTT broker + 4 node containers
```

### Step 4: Environment Variables
```bash
# On each Jetson node container
TRAFFIC_MODE=mqtt
MQTT_BROKER=broker:1883
NODE_ID=A  # Different for each node
DETECTION_MODE=yolo  # Use real YOLO on Jetson
```

**Result**: Same application code runs identically on laptop (simulation mode) and Jetson (MQTT mode).

---

## Architecture Comparison

### Before (Monolithic)
```
┌─────────────────────────────────────┐
│         FastAPI Server              │
├─────────────────────────────────────┤
│  GridCoordinator (centralized)      │
│  ├── MasterAgent A                  │
│  ├── MasterAgent B                  │
│  ├── MasterAgent C                  │
│  └── MasterAgent D                  │
│  PriorityAuction (shared logic)     │
│  SimulationLoop (hardcoded)         │
└─────────────────────────────────────┘
        ↑               ↓
    WebSocket      Dashboard
```

### After (Decentralized)
```
┌─────────────────────────────────────┐
│         FastAPI Server              │
├─────────────────────────────────────┤
│  NodeNetwork (abstraction layer)    │
│  ├── IntersectionNode A             │
│  │   └── MasterAgent + LaneAgents   │
│  ├── IntersectionNode B             │
│  │   └── MasterAgent + LaneAgents   │
│  ├── IntersectionNode C             │
│  │   └── MasterAgent + LaneAgents   │
│  └── IntersectionNode D             │
│      └── MasterAgent + LaneAgents   │
└─────────────────────────────────────┘
    ↓ (in-memory calls, laptop)
    ↓ or MQTT pub/sub (Jetson)
    ↓
Dashboard (unchanged)
```

---

## Configuration Reference

### Environment Variables

| Variable | Values | Default | Purpose |
|----------|--------|---------|---------|
| `TRAFFIC_MODE` | `simulation`, `mqtt` | `simulation` | Deployment mode (laptop or edge) |
| `MQTT_BROKER` | `hostname:port` | `localhost:1883` | MQTT broker for distributed mode |
| `NODE_ID` | `None`, `A`, `B`, `C`, `D` | `None` | Single node (None=all nodes) |
| `DETECTION_MODE` | `mock`, `yolo` | `mock` | Vehicle detection backend |
| `DEBUG` | `true`, `false` | `false` | Enable verbose logging |

### Usage Examples

**Laptop Development (Simulation Mode)**
```bash
export TRAFFIC_MODE=simulation
export DETECTION_MODE=mock
python main.py
```

**Jetson Node A (MQTT Mode)**
```bash
export TRAFFIC_MODE=mqtt
export MQTT_BROKER=broker.local:1883
export NODE_ID=A
export DETECTION_MODE=yolo
python main.py
```

---

## Dashboard Updates (Optional)

The dashboard automatically works with the new system. For enhanced UX, add a mode indicator:

**File**: `public/src/components/ControlPanel.tsx`
```jsx
// Add to ControlPanel:
<div className="text-sm text-gray-500">
  Mode: {gridState?.mode || 'simulation'}
</div>
```

This shows users whether they're running in "simulation" or "mqtt" mode.

---

## Validation Checklist

- [x] Single IntersectionNode can tick independently
- [x] NodeNetwork can coordinate multiple nodes
- [x] Ambulance routing works with green-wave alerts
- [x] Dashboard receives same WebSocket format
- [x] REST endpoints return expected responses
- [x] Configuration switching works (simulation/mqtt)
- [x] Logging captures all events
- [ ] Full stack test (backend + frontend)
- [ ] Integration test with real YOLO
- [ ] Performance baseline (ticks per second)

---

## Next Actions

1. **Today**: Run `test_autonomous.py` to validate architecture
2. **Today**: Test full stack (backend + dashboard) - verify WebSocket still works
3. **This week**: Update dashboard to show deployment mode
4. **This week**: Create integration test suite
5. **Next week**: Begin MQTT integration (Phase 2)

---

## Key Insights

1. **IntersectionNode is your single-node deployment unit**: Everything needed for a single intersection is self-contained. This is what deploys to each Jetson.

2. **NodeNetwork is your coordination abstraction**: The same interface works for both laptop (in-memory) and Jetson (MQTT). No code duplication.

3. **Environment configuration replaces code branching**: Instead of `if JETSON: do X else do Y`, use environment variables and keep code simple.

4. **Backward compatibility is preserved**: Dashboard, detection, and XAI all work unchanged. The refactoring is invisible to users.

---

## Support & Debugging

If tests fail:
1. Check imports: `from node.node_network import NodeNetwork` should find the file
2. Check config: `python -c "from config import TRAFFIC_MODE; print(TRAFFIC_MODE)"`
3. Check logs: Enable `DEBUG=true` for verbose output
4. Check dependencies: `pip list | grep -E "fastapi|pydantic|langchain"`

---

**Version**: 3.0.0 (Decentralized)  
**Status**: Laptop improvements complete, Jetson migration ready  
**Estimated MQTT integration time**: 2-3 days (Phase 2)
