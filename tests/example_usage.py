#!/usr/bin/env python3
"""
Example usage of the improved Computer Use Agent with OutputState.

This demonstrates how to use the agent and get structured results.
"""

import asyncio
from agent.graph import workflow_graph
from agent.state import GraphInput, OutputState

async def run_agent_example(user_request: str) -> OutputState:
    """
    Run the Computer Use Agent with a user request and return structured results.
    
    Args:
        user_request: The task for the agent to perform
        
    Returns:
        OutputState: Structured result with answer, execution metadata, and status
    """
    print(f"🚀 Starting Computer Use Agent...")
    print(f"📝 User Request: {user_request}")
    print("-" * 60)
    
    try:
        # Create input for the graph
        input_data = GraphInput(user_request=user_request)
        
        # Run the workflow - this now returns an OutputState
        result = await workflow_graph.ainvoke(input_data)
        
        return result
        
    except Exception as e:
        print(f"❌ Error running agent: {e}")
        # Return a failed output state
        return OutputState(
            status="failed",
            answer=f"Agent execution failed: {str(e)}",
            user_request=user_request,
            error_message=str(e)
        )

def display_result(result: OutputState):
    """Display the result in a user-friendly format."""
    print("\n" + "=" * 60)
    print("📊 EXECUTION RESULT")
    print("=" * 60)
    
    if result.task_completed_successfully:
        print("✅ Status: SUCCESS")
        print(f"💡 Answer: {result.answer}")
    else:
        print("❌ Status: FAILED")
        print(f"🚨 Error: {result.error_message}")
    
    print(f"\n📈 Execution Summary:")
    print(f"   • Total Interactions: {result.total_interactions}")
    print(f"   • Execution Time: {result.execution_time_seconds:.1f} seconds")
    print(f"   • Screenshots Taken: {result.screenshots_taken}")
    
    if result.actions_performed:
        print(f"   • Recent Actions: {' → '.join(result.actions_performed[-3:])}")
    
    print(f"\n🎯 Original Request: {result.user_request}")
    print("=" * 60)

async def main():
    """Main function demonstrating different use cases."""
    
    print("🤖 Computer Use Agent - OutputState Example")
    print("=" * 60)
    
    # Example 1: Search task
    print("\n📋 Example 1: Search Task")
    result1 = await run_agent_example("find top 10 food dishes in India")
    display_result(result1)
    
    # Example 2: Navigation task
    print("\n📋 Example 2: Navigation Task")
    result2 = await run_agent_example("go to google.com and search for python tutorials")
    display_result(result2)
    
    # Example 3: Information gathering
    print("\n📋 Example 3: Information Gathering")
    result3 = await run_agent_example("find information about machine learning algorithms")
    display_result(result3)
    
    # Show how to programmatically handle results
    print("\n🔧 Programmatic Usage Examples:")
    print("-" * 40)
    
    # Example of checking success and extracting answer
    if result1.task_completed_successfully:
        print(f"✅ Task 1 succeeded: {result1.answer[:100]}...")
    else:
        print(f"❌ Task 1 failed: {result1.error_message}")
    
    # Example of checking execution time
    if result1.execution_time_seconds > 30:
        print(f"⏰ Task 1 took longer than expected: {result1.execution_time_seconds:.1f}s")
    else:
        print(f"⚡ Task 1 completed quickly: {result1.execution_time_seconds:.1f}s")
    
    # Example of checking interaction count
    if result1.total_interactions > 10:
        print(f"🔄 Task 1 required many interactions: {result1.total_interactions}")
    else:
        print(f"🎯 Task 1 was efficient: {result1.total_interactions} interactions")

if __name__ == "__main__":
    print("This example demonstrates the improved Computer Use Agent with OutputState.")
    print("Key improvements:")
    print("✅ Structured output with clear success/failure indication")
    print("✅ Detailed execution metadata (time, interactions, screenshots)")
    print("✅ Formatted output for easy display")
    print("✅ Separate answer field for the actual result")
    print("✅ Better loop prevention and completion detection")
    print("✅ Error handling with specific error messages")
    print("\nTo run this example with actual execution:")
    print("1. Set up your API keys (OPENAI_API_KEY, E2B_API_KEY)")
    print("2. Uncomment the following line:")
    print("# asyncio.run(main())")
    print("\nFor now, this is a demonstration of the structure and capabilities.")
    
    # Uncomment to run actual examples (requires API keys):
    # asyncio.run(main()) 