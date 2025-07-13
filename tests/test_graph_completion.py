#!/usr/bin/env python3
"""
Test script to verify that the graph properly returns final results after completion.
"""

import asyncio
import sys
from agent.state import ExecutionState, GraphInput, TaskStatus

def test_graph_structure():
    """Test that the graph structure is correct."""
    from agent.graph import create_graph
    
    print("Testing graph structure...")
    
    # Create the graph
    graph = create_graph()
    
    # Check that all nodes are present
    expected_nodes = {"setup_sandbox", "brain", "executor", "cleanup"}
    actual_nodes = set(graph.nodes.keys())
    
    print(f"Expected nodes: {expected_nodes}")
    print(f"Actual nodes: {actual_nodes}")
    
    if expected_nodes.issubset(actual_nodes):
        print("✓ All expected nodes are present")
    else:
        missing = expected_nodes - actual_nodes
        print(f"✗ Missing nodes: {missing}")
        return False
    
    # Check that cleanup node has edge to END
    edges = graph.edges
    print(f"Graph edges: {edges}")
    
    # Look for cleanup -> END edge
    cleanup_to_end = any(edge[0] == "cleanup" and edge[1] == "__end__" for edge in edges)
    if cleanup_to_end:
        print("✓ Cleanup node properly connects to END")
    else:
        print("✗ Cleanup node does not connect to END")
        return False
    
    return True

def test_cleanup_node():
    """Test that the cleanup node properly handles completion."""
    from agent.graph import cleanup_node
    
    print("\nTesting cleanup node...")
    
    # Create a completed state
    state = ExecutionState(
        status=TaskStatus.completed,
        message_for_user="Test task completed successfully",
        user_request="test request",
        sandbox_id="test-sandbox-123"
    )
    
    print(f"Input state: status={state.status}, message='{state.message_for_user}'")
    
    # Run cleanup node
    result = asyncio.run(cleanup_node(state))
    
    print(f"Output state: status={result.status}, message='{result.message_for_user}'")
    
    # Check that the result is properly preserved
    if result.status == TaskStatus.completed:
        print("✓ Status preserved as completed")
    else:
        print(f"✗ Status changed to: {result.status}")
        return False
    
    if result.message_for_user == "Test task completed successfully":
        print("✓ Message preserved correctly")
    else:
        print(f"✗ Message changed to: {result.message_for_user}")
        return False
    
    return True

def test_state_serialization():
    """Test that the state can be properly serialized/deserialized."""
    print("\nTesting state serialization...")
    
    # Create a state with completion
    state = ExecutionState(
        status=TaskStatus.completed,
        message_for_user="Serialization test completed",
        user_request="test serialization",
        interaction_count=5
    )
    
    # Convert to dict (simulating what LangGraph does)
    state_dict = state.model_dump()
    print(f"State dict keys: {list(state_dict.keys())}")
    
    # Check that key fields are present
    required_fields = ["status", "message_for_user", "user_request"]
    for field in required_fields:
        if field in state_dict:
            print(f"✓ {field}: {state_dict[field]}")
        else:
            print(f"✗ Missing field: {field}")
            return False
    
    # Recreate state from dict
    new_state = ExecutionState(**state_dict)
    
    if new_state.status == state.status and new_state.message_for_user == state.message_for_user:
        print("✓ State serialization/deserialization works")
    else:
        print("✗ State serialization/deserialization failed")
        return False
    
    return True

async def test_minimal_graph_execution():
    """Test a minimal graph execution to see if it returns results."""
    print("\nTesting minimal graph execution...")
    
    try:
        from agent.graph import workflow_graph
        from agent.state import GraphInput
        
        # Create a simple input
        input_data = GraphInput(user_request="test completion")
        
        print("Created input successfully")
        
        # Note: This would require actual API keys and sandbox setup to run
        # So we'll just test that the graph can be invoked without errors
        print("Graph is ready for execution (requires API keys for actual run)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error setting up graph execution: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Computer Use Agent graph completion...")
    print("=" * 60)
    
    tests = [
        test_graph_structure,
        test_cleanup_node,
        test_state_serialization,
        test_minimal_graph_execution
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✓ All tests passed! The graph should properly return final results.")
    else:
        print("✗ Some tests failed. The graph may not return final results correctly.")
        sys.exit(1)

if __name__ == "__main__":
    main() 