#!/usr/bin/env python3
"""
Test script to verify sandbox reuse logic handles cleaned up sandboxes correctly.
"""

import asyncio
from agent.state import ExecutionState, TaskStatus
from agent.graph import setup_sandbox, cleanup_sandbox
from agent.cua.sandbox import get_desktop

async def test_sandbox_verification():
    """Test that sandbox verification works correctly."""
    print("Testing sandbox verification logic...")
    
    # Create initial state
    state = ExecutionState(
        user_request="test request",
        sandbox_id="",
        sandbox_url=""
    )
    
    # First setup should create a new sandbox
    print("\n1. First setup - should create new sandbox")
    config = {"configurable": {}}
    state = await setup_sandbox(state, config=config)
    
    print(f"   Created sandbox: {state.sandbox_id}")
    print(f"   Sandbox URL: {state.sandbox_url}")
    print(f"   Status: {state.status}")
    
    # Verify sandbox is active
    desktop = get_desktop(state.sandbox_id)
    if desktop:
        print("   ✓ Sandbox is active and accessible")
    else:
        print("   ✗ Sandbox is not accessible")
        return False
    
    # Save sandbox info for reuse test
    sandbox_id = state.sandbox_id
    sandbox_url = state.sandbox_url
    
    # Second setup with same sandbox should reuse
    print("\n2. Second setup - should reuse existing sandbox")
    state2 = ExecutionState(
        user_request="test request 2",
        sandbox_id=sandbox_id,
        sandbox_url=sandbox_url
    )
    
    state2 = await setup_sandbox(state2, config=config)
    
    if state2.sandbox_id == sandbox_id:
        print(f"   ✓ Reused existing sandbox: {state2.sandbox_id}")
    else:
        print(f"   ✗ Created new sandbox instead of reusing: {state2.sandbox_id}")
        return False
    
    # Clean up the sandbox
    print("\n3. Cleaning up sandbox...")
    cleanup_sandbox(sandbox_id, graceful=False)
    print(f"   Cleaned up sandbox: {sandbox_id}")
    
    # Verify sandbox is no longer active
    desktop = get_desktop(sandbox_id)
    if not desktop:
        print("   ✓ Sandbox is no longer accessible after cleanup")
    else:
        print("   ✗ Sandbox is still accessible after cleanup")
    
    # Third setup with cleaned up sandbox should create new one
    print("\n4. Third setup - should create new sandbox (old one cleaned up)")
    state3 = ExecutionState(
        user_request="test request 3",
        sandbox_id=sandbox_id,  # Old sandbox ID
        sandbox_url=sandbox_url  # Old sandbox URL
    )
    
    state3 = await setup_sandbox(state3, config=config)
    
    if state3.sandbox_id != sandbox_id:
        print(f"   ✓ Created new sandbox: {state3.sandbox_id}")
        print("   ✓ Correctly detected old sandbox was cleaned up")
    else:
        print(f"   ✗ Attempted to reuse cleaned up sandbox: {state3.sandbox_id}")
        return False
    
    # Clean up the new sandbox
    cleanup_sandbox(state3.sandbox_id, graceful=False)
    print(f"   Cleaned up new sandbox: {state3.sandbox_id}")
    
    return True

async def test_state_reset():
    """Test that state is properly reset between executions."""
    print("\nTesting state reset between executions...")
    
    # Create state with some existing data
    state = ExecutionState(
        user_request="old request",
        sandbox_id="old-sandbox",
        sandbox_url="old-url",
        status=TaskStatus.completed,
        interaction_count=5,
        message_for_user="old message",
        app_launched=True
    )
    
    print(f"   Before setup: status={state.status}, interactions={state.interaction_count}")
    print(f"   Before setup: app_launched={state.app_launched}")
    
    # Setup should reset the state
    config = {"configurable": {}}
    state = await setup_sandbox(state, config=config)
    
    print(f"   After setup: status={state.status}, interactions={state.interaction_count}")
    print(f"   After setup: app_launched={state.app_launched}")
    print(f"   After setup: message_for_user='{state.message_for_user}'")
    
    # Verify state was reset
    if (state.status == TaskStatus.in_progress and 
        state.interaction_count == 0 and 
        state.message_for_user == "" and 
        state.app_launched == False):
        print("   ✓ State was properly reset")
        
        # Clean up
        cleanup_sandbox(state.sandbox_id, graceful=False)
        return True
    else:
        print("   ✗ State was not properly reset")
        return False

def main():
    """Run all tests."""
    print("Testing Sandbox Reuse and State Reset")
    print("=" * 50)
    
    tests = [
        test_sandbox_verification,
        test_state_reset
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if asyncio.run(test()):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✓ All tests passed! Sandbox reuse logic is working correctly.")
        print("\nKey fixes implemented:")
        print("- Verifies sandbox is active before reusing")
        print("- Creates new sandbox if old one is cleaned up")
        print("- Properly resets state between executions")
        print("- Handles sandbox cleanup gracefully")
    else:
        print("✗ Some tests failed. Sandbox reuse logic needs fixes.")

if __name__ == "__main__":
    print("This test verifies that sandbox reuse works correctly.")
    print("Note: This test requires actual API keys and will create/cleanup sandboxes.")
    print("To run this test:")
    print("1. Set up your API keys (OPENAI_API_KEY, E2B_API_KEY)")
    print("2. Uncomment the following line:")
    print("# main()")
    print("\nFor now, this demonstrates the testing approach.")
    
    # Uncomment to run actual tests (requires API keys):
    # main() 