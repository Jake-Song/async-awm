import asyncio
from agent_world_model_env import AWMEnv
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

async def main():
    async with AWMEnv(base_url="http://localhost:8899") as env:
        # Reset to a scenario with a specific task
        result = await env.reset(scenario="e_commerce_33", task_idx=0)
        print(f"Task: {result.observation.task}")
        print(f"Tools available: {result.observation.num_tools}")
        print(f"Verifier support: {result.observation.has_verifier}")  # {sql: True, code: True}

        # List available tools
        tools = await env.list_tools()
        for tool in tools[:3]:
            print(f"  - {tool.name}: {tool.description}")

        # Call a tool
        obs = await env.call_tool("search_products", query="headphones")
        print(f"Result: {obs.tool_result}")

        # Run verification (can be called multiple times with different modes)
        result = await env.step(CallToolAction(
            tool_name="verify",
            arguments={"verifier_mode": "code", "final_answer": "optional answer"}
        ))
        print(f"Reward type: {result.observation.reward_type}")
        print(f"Reward: {result.reward}")
        print(f"Verify result: {result.observation.verify_result}")

        # End episode (destroys subprocess; set keep_session=True to preserve files)
        result = await env.step(CallToolAction(tool_name="done", arguments={"keep_session": False}))
        print(f"Episode done: {result.done}")

asyncio.run(main())