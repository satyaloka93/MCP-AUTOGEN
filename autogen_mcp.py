import asyncio
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
#pip install opentelemetry-instrumentation-openai
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
#######################################################################
from phoenix.otel import register
import phoenix as px
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
#px.launch_app()
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
# configure the Phoenix tracer
endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider = register(
  project_name="my-llm-app", # Default is 'default'
  auto_instrument=True # Auto-instrument your app based on installed OI dependencies
)

OpenAIInstrumentor().instrument()
######################################################################
# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Run AutoGen MCP example with specified LLM.")
parser.add_argument(
    "--llm",
    type=str,
    default="local",
    choices=["local", "gemini"],
    help="Choose the LLM to use: 'local' or 'gemini'.",
)
args = parser.parse_args()

# --- Load Environment Variables ---
load_dotenv()
brave_api_key = os.getenv("BRAVE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not brave_api_key:
    raise ValueError("BRAVE_API_KEY not found in environment or .env file.")
if args.llm == "gemini" and not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found, but --llm gemini was specified.")
elif not gemini_api_key:
     print("Warning: GEMINI_API_KEY not found. Gemini model option will not work unless specified.")


# --- Conditional Model Client Setup ---
model_client_config = {
    "max_tokens": 4096,
    "temperature": 1.0,
    "model_capabilities": {
        "function_calling": True,
        "vision": True, # Keep vision capability for potential future use
        "json_output": True,
    },
}

if args.llm == "local":
    print("Using local LLM configuration.")
    model_client_config.update({
        "model": "local", # Or specific local model name if needed
        "base_url": "http://localhost:8080/v1",
        "api_key": "sk-xxxxxx", # Placeholder for local server
        "top_k": 64, # Local specific params
        "repeat_penalty": 1.0, # Local specific params
    })
elif args.llm == "gemini":
    print("Using Gemini configuration.")
    model_client_config.update({
        "model": "gemini-2.5-pro-exp-03-25",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": gemini_api_key,
        # Gemini might not use top_k or repeat_penalty, remove if they cause errors
    })

model_client = OpenAIChatCompletionClient(**model_client_config)


# --- MCP Tool Server Params ---
# Using direct key embedding for simplicity, ensure BRAVE_API_KEY is set
docker_args = ["run", "-i", "--rm", "-e", f"BRAVE_API_KEY={brave_api_key}", "mcp/brave-search"]
fetch_args = ["mcp-server-fetch"]

brave_search_params = StdioServerParams(command="docker", args=docker_args)
fetch_params = StdioServerParams(command="/home/gthom/miniconda3/envs/jupyter/bin/uvx", args=fetch_args)

async def main() -> None:
    # --- Get Tools from MCP Servers ---
    print("Connecting to MCP servers...")
    try:
        brave_tools = await mcp_server_tools(brave_search_params)
        print(f"Brave tools retrieved: {[adapter.name for adapter in brave_tools]}")
        fetch_tools = await mcp_server_tools(fetch_params)
        print(f"Fetch tools retrieved: {[adapter.name for adapter in fetch_tools]}")
        all_tools = brave_tools + fetch_tools
        if not all_tools:
             print("Warning: No tools were loaded from MCP servers.")
             return # Exit if no tools loaded
        print("MCP tools loaded successfully.")
    except Exception as e:
        print(f"Error connecting to MCP servers or getting tools: {e}")
        return # Exit if connection fails

    # --- Define Agents ---
    assistant = AssistantAgent(
        name="Assistant",
        model_client=model_client, # Use the configured client
        tools=all_tools,
        reflect_on_tool_use=False, # Keep False based on previous attempts
        system_message="You are a helpful assistant. Execute the user's request step-by-step. Use the available tools.",
    )

    # Generic Critic System Message
    critic_system_message = """You are a critic. Your role is to examine the conversation history and the original user task provided to the Assistant, stating TERMINATE when done.

Review the Assistant's progress towards completing the user's task:
1.  Is the Assistant understanding the user's request?
2.  Is the Assistant selecting appropriate tools (if any are needed) from the available list to achieve the task steps?
3.  Is the Assistant executing the steps logically?
4.  Has the Assistant successfully completed all parts of the user's request?

Analyze the outcome:
- If the Assistant has successfully completed the entire task, respond ONLY with the word TERMINATE.
- If the Assistant is stuck, has missed a step, or is using a tool incorrectly, provide specific, constructive feedback to guide it towards the next logical step based on the user's goal.
- If the Assistant encounters a tool execution error:
    - Acknowledge the specific error.
    - Suggest a potential alternative approach (e.g., trying a different tool, modifying parameters) if one seems feasible based on the task and available tools.
    - If the error prevents task completion and no alternatives exist, state this clearly and respond ONLY with the word TERMINATE.
- If the Assistant seems to be hallucinating or deviating significantly from the task, gently redirect it back to the user's original request.

Do not try to complete the task yourself. Focus solely on evaluating the Assistant's performance and providing guidance to help it fulfill the user's request effectively using the provided tools."""

    critic = AssistantAgent(
        name="Critic",
        model_client=model_client, # Use the configured client
        system_message=critic_system_message,
    )

    # --- Define Team ---
    termination_condition = TextMentionTermination("TERMINATE")

    team = RoundRobinGroupChat(
        participants=[assistant, critic],
        termination_condition=termination_condition,
        max_turns=15, # Increased max_turns slightly to allow for retries
    )

    # --- Run Task ---
    task = input("Please enter the task for the Assistant: ")
    print(f"\n--- Starting Task --- \n{task}\n---------------------\n")

    await Console(team.run_stream(task=task))

    print("\n--- Task Finished ---")

if __name__ == "__main__":
    try:
        asyncio.run(main())
        #time.sleep(500)
    except Exception as e:
        print(f"An error occurred during execution: {e}")