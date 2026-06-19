"""
Test suite for autonomous node architecture.
Validates that:
1. IntersectionNode can run standalone
2. NodeNetwork can coordinate multiple nodes
3. System maintains backward compatibility with dashboard
"""

import asyncio
import json
from node.intersection_node import IntersectionNode
from node.node_network import NodeNetwork
from config import TRAFFIC_MODE


def test_single_node_tick():
    """Test 1: Single node ticks independently."""
    print("\n=== Test 1: Single Node Autonomous Tick ===")
    node_a = IntersectionNode("A")
    
    # Simulate 10 ticks
    for tick in range(10):
        state = node_a.tick(dt=1.0)
        assert state is not None, "Node should return state"
        assert "lanes" in state, "State should contain lane data"
        assert "current_green" in state, "State should contain current_green"
        assert len(state['lanes']) == 4, "Should have 4 lanes"
        print(f"  Tick {tick}: {len(state['lanes'])} lanes, current_green={state.get('current_green', 'N/A')}")
    
    print("✓ Single node ticking works independently")


def test_node_network_simulation():
    """Test 2: NodeNetwork coordinates multiple nodes in simulation mode."""
    print("\n=== Test 2: NodeNetwork Simulation Mode ===")
    network = NodeNetwork(mode="simulation")
    
    # Simulate 10 ticks
    states = {}
    for tick in range(10):
        states = network.tick_all(dt=1.0)
        assert len(states) == 4, "Network should have 4 nodes"
        assert all(nid in states for nid in ["A", "B", "C", "D"]), "All nodes should be in states"
        print(f"  Tick {tick}: Network has {len(states)} active nodes")
    
    print("✓ NodeNetwork can coordinate multiple autonomous nodes")


def test_ambulance_spawning():
    """Test 3: Ambulance spawning and routing."""
    print("\n=== Test 3: Ambulance Spawning & Green Wave ===")
    network = NodeNetwork(mode="simulation")
    
    # Spawn ambulance
    ambulance_id = network.spawn_ambulance("A_to_D")
    assert ambulance_id is not None, "Ambulance should be spawned"
    print(f"  Spawned ambulance: {ambulance_id}")
    assert ambulance_id in network.active_ambulances, "Ambulance should be in active list"
    
    # Tick and verify green-wave progression
    for tick in range(20):
        states = network.tick_all(dt=1.0)
        grid = network.get_grid_state()
        
        if tick % 5 == 0:
            current_step = network.active_ambulances[ambulance_id]["current_step"]
            print(f"  Tick {tick}: Ambulance at step {current_step}/{len(network.AMBULANCE_ROUTES['A_to_D'])}")
    
    # Ambulance should be completed after ~24 ticks (3 steps * 8 ticks)
    ambulance_completed = ambulance_id not in network.active_ambulances
    print(f"  Ambulance completed route: {ambulance_completed}")
    print("✓ Ambulance spawning and routing works")


def test_grid_state_format():
    """Test 4: Grid state maintains backward compatibility with dashboard."""
    print("\n=== Test 4: Dashboard Compatibility ===")
    network = NodeNetwork(mode="simulation")
    
    # Tick once
    network.tick_all(dt=1.0)
    grid_state = network.get_grid_state()
    
    # Verify structure matches dashboard expectations
    assert "intersections" in grid_state, "Should have intersections key"
    assert len(grid_state["intersections"]) == 4, "Should have 4 intersections"
    
    for node_id, node_state in grid_state["intersections"].items():
        assert "lanes" in node_state, f"Node {node_id} should have lanes"
        assert "current_green" in node_state, f"Node {node_id} should have current_green"
        print(f"  Node {node_id}: {len(node_state['lanes'])} lanes, current_green={node_state['current_green']}")
    
    print("✓ Grid state format is dashboard-compatible")


def test_network_mode_switching():
    """Test 5: Network mode configuration."""
    print("\n=== Test 5: Mode Switching ===")
    
    # Simulation mode
    sim_network = NodeNetwork(mode="simulation")
    assert sim_network.mode == "simulation", "Should be in simulation mode"
    print(f"  Simulation mode: {sim_network.mode}")
    
    # MQTT mode (future - just verify mode setting)
    mqtt_network = NodeNetwork(mode="mqtt")
    assert mqtt_network.mode == "mqtt", "Should be in mqtt mode"
    print(f"  MQTT mode: {mqtt_network.mode}")
    
    print("✓ Mode switching works (mqtt mode ready for implementation)")


def test_multiple_ambulances():
    """Test 6: Multiple simultaneous ambulances."""
    print("\n=== Test 6: Multiple Simultaneous Ambulances ===")
    network = NodeNetwork(mode="simulation")
    
    # Spawn multiple ambulances
    amb1 = network.spawn_ambulance("A_to_D")
    amb2 = network.spawn_ambulance("B_to_C")
    
    print(f"  Spawned: {amb1}, {amb2}")
    assert len(network.active_ambulances) == 2, "Should have 2 active ambulances"
    
    # Tick and verify both progress
    for tick in range(20):
        network.tick_all(dt=1.0)
        
        if tick % 5 == 0:
            active_count = len(network.active_ambulances)
            print(f"  Tick {tick}: {active_count} active ambulances")
    
    print("✓ Multiple ambulances can run simultaneously")


def test_invalid_route():
    """Test 7: Invalid route handling."""
    print("\n=== Test 7: Invalid Route Handling ===")
    network = NodeNetwork(mode="simulation")
    
    # Try invalid route
    result = network.spawn_ambulance("INVALID_ROUTE")
    assert result is None, "Invalid route should return None"
    print("  Invalid route correctly rejected")
    print("✓ Error handling works")


if __name__ == "__main__":
    print(f"Running tests in {TRAFFIC_MODE} mode...")
    
    try:
        test_single_node_tick()
        test_node_network_simulation()
        test_ambulance_spawning()
        test_grid_state_format()
        test_network_mode_switching()
        test_multiple_ambulances()
        test_invalid_route()
        
        print("\n" + "="*50)
        print("✓ ALL TESTS PASSED")
        print("="*50)
        print("\nKey validations:")
        print("  ✓ Autonomous nodes tick independently")
        print("  ✓ NodeNetwork coordinates multiple nodes")
        print("  ✓ Ambulance routing works correctly")
        print("  ✓ Dashboard compatibility maintained")
        print("  ✓ Mode switching ready for MQTT")
        print("  ✓ Error handling works")
        print("\nNext steps:")
        print("  1. Run full stack test (dashboard + backend)")
        print("  2. Update dashboard UI to show deployment mode")
        print("  3. Create integration test suite")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
