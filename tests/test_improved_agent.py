#!/usr/bin/env python3
"""
Test script to demonstrate improved agent behavior with better completion detection.
"""

import asyncio
from agent.graph import workflow_graph
from agent.state import GraphInput

async def test_improved_agent():
    """Test the improved agent with better completion detection."""
    
    print("Testing improved Computer Use Agent with better completion detection...")
    print("=" * 70)
    
    # Test case: Search for top 10 Indian food dishes
    test_request = "find top 10 food dishes in India"
    
    print(f"User Request: {test_request}")
    print("-" * 50)
    
    try:
        # Create input
        input_data = GraphInput(user_request=test_request)
        
        # Run the workflow with a reasonable recursion limit
        config = {"recursion_limit": 25}
        
        print("Starting agent execution...")
        result = await workflow_graph.ainvoke(input_data, config=config)
        
        print("\nAgent execution completed!")
        print(f"Final Status: {result.status}")
        print(f"Final Message: {result.message_for_user}")
        print(f"Total Interactions: {result.interaction_count}")
        
        # Show recent actions
        if result.action_history:
            print(f"\nRecent Actions: {' -> '.join(result.action_history[-5:])}")
        
        return result
        
    except Exception as e:
        print(f"Error during execution: {e}")
        return None

if __name__ == "__main__":
    print("This script demonstrates the improved agent behavior.")
    print("Key improvements:")
    print("1. Better completion detection in brain prompts")
    print("2. Aggressive loop breaking with forced completion")
    print("3. Intelligent completion based on request patterns")
    print("4. Reduced iteration limit to prevent infinite loops")
    print("5. Enhanced repeated action handling")
    print("\nNote: This is a demonstration script. To actually run the agent,")
    print("you would need valid API keys and environment setup.")
    print("\nTo test with actual execution, uncomment the following line:")
    print("# asyncio.run(test_improved_agent())") 