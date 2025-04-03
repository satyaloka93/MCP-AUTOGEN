# pip install openinference-instrumentation-openai-agents openai-agents
# pip install openinference-instrumentation-haystack haystack-ai opentelemetry-sdk opentelemetry-exporter-otlp arize-phoenix
# NOTE: No Pipeline or experimental components needed now.
# Run python -m phoenix.server.main serve
#


import os
import time
import re # Re-added for URL extraction
from dotenv import load_dotenv
# No Pipeline, ConditionalRouter, or custom components needed
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.tools import ToolInvoker
# Correct import: Add ChatRole, ToolCall, ToolCallResult
from haystack.dataclasses import ChatMessage, ChatRole, ToolCall, ToolCallResult
# Import asdict for converting ToolCall dataclass
from dataclasses import asdict
# Import ToolInvocationError for specific handling
from haystack.tools.errors import ToolInvocationError

# Do not change the import below, the documentation is incorrect
from haystack_integrations.tools.mcp import MCPTool, SSEServerInfo, StdioServerInfo
from typing import List, Dict, Any, Optional
from haystack.utils import Secret


########################
# Tracing setup remains the same How did Elon musk react to the Wisconsin Supreme court race
from openinference.instrumentation.haystack import HaystackInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

HaystackInstrumentor().instrument(tracer_provider=tracer_provider)
time.sleep(5)
#########################

# Env var and API key loading remains the same
load_dotenv()
brave_api_key = os.getenv("BRAVE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not brave_api_key:
    raise ValueError("BRAVE_API_KEY not found in environment or .env file.")
if not gemini_api_key:
    print("Warning: GEMINI_API_KEY not found. Gemini model option will not work.")

# --- MCP Tool Setup remains the same ---
docker_args = ["run", "-i", "--rm", "-e", "BRAVE_API_KEY", "mcp/brave-search"]
docker_env = {"BRAVE_API_KEY": brave_api_key}
fetch_args = ["mcp-server-fetch"]
fetch_server_info = StdioServerInfo(command="/home/gthom/miniconda3/envs/jupyter/bin/uvx", args=fetch_args)
brave_server_info = StdioServerInfo(command="docker", args=docker_args, env=docker_env)

brave_search_tool = MCPTool(
    name="brave_web_search",
    server_info=brave_server_info,
    description="Performs a web search using the Brave Search API when you need current information or web results. Returns search results including URLs that can potentially be used with the 'fetch' tool to get full page content."
)

fetch_tool = MCPTool(
    name="fetch",
    server_info=fetch_server_info,
    description="Fetches the full content of a specific URL (e.g., one identified by 'brave_web_search') and extracts its contents as markdown. Use this when you need the complete text of a webpage, not just a summary or search snippet."
)

# --- Haystack Component Setup ---

# LLM Choice and Configuration remains the same
llm_choice = ""
while llm_choice not in ["local", "gemini"]:
    llm_choice = input("Choose LLM ('local' or 'gemini'): ").lower().strip()

# ... (LLM configuration based on choice) ...
if llm_choice == "local":
    print("Using local LLM configuration.")
    api_base_url = "http://127.0.0.1:8080/v1"
    model_name = 'gpt-3.5-turbo-0613'
    api_key = Secret.from_token("dummy-unused-key")
elif llm_choice == "gemini":
    if not gemini_api_key:
        raise ValueError("Cannot use Gemini: GEMINI_API_KEY not found.")
    print("Using Gemini configuration.")
    api_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    model_name = 'gemini-2.5-pro-exp-03-25'
    api_key = Secret.from_token(gemini_api_key)

# Initialize the LLM generator (needs tools defined for tool calling)
llm = OpenAIChatGenerator(
    api_base_url=api_base_url,
    model=model_name,
    api_key=api_key,
    tools=[brave_search_tool, fetch_tool]
)

# Initialize ToolInvoker
tool_invoker = ToolInvoker(tools=[brave_search_tool, fetch_tool])

# --- Manual Agent Interaction Loop ---
print("\nChatting using manual loop with Haystack components (type 'quit' to exit):")
# Stores the conversation history across turns
conversation_history: List[ChatMessage] = []

# System Prompt (kept concise as complex logic is now partly manual)
system_prompt_text = """
You are a helpful assistant. Use 'brave_web_search' to find information and URLs. Use 'fetch' to get content from a URL when requested or necessary.
If a tool fails, inform the user. Base your final response on the available information, including fetched content if successful.
"""

MAX_TOOL_CALL_LOOPS = 5 # Safety limit for inner loop

# Outer loop for user interaction
while True:
    # Store original user input for fetch check
    original_user_input = input("You: ")
    if original_user_input.lower() == 'quit':
        print("Exiting chat.")
        break

    # Prepare messages for the *start* of this turn
    user_message = ChatMessage.from_user(original_user_input)
    # Create a copy for the current turn to avoid modifying persistent history directly
    current_turn_messages = list(conversation_history) + [user_message]

    # Add system prompt if not present
    if not any(msg.role == ChatRole.SYSTEM for msg in current_turn_messages):
         current_turn_messages.insert(0, ChatMessage.from_system(system_prompt_text))

    inner_loop_count = 0
    final_reply_for_turn = None
    force_fetch_url: Optional[str] = None # URL to fetch if triggered manually
    # Store the index where this turn's messages begin in current_turn_messages
    # This helps later to correctly update the persistent conversation_history
    turn_start_index_in_messages = len(current_turn_messages) -1 # Index of user_message

    # Inner loop for LLM calls and tool executions within a single turn
    while inner_loop_count < MAX_TOOL_CALL_LOOPS:
        inner_loop_count += 1
        print(f"\nAgent thinking (Turn Cycle {inner_loop_count})...")

        llm_reply = None # Reset llm_reply for the cycle
        tool_messages = [] # Reset tool messages for the cycle

        # --- Check if we need to manually trigger fetch ---
        if force_fetch_url:
            print(f"Manually triggering fetch for: {force_fetch_url}")
            manual_fetch_call = ToolCall(tool_name="fetch", arguments={"url": force_fetch_url})
            # Create a message specifically for this manual call
            llm_reply = ChatMessage.from_assistant(tool_calls=[manual_fetch_call])
            current_turn_messages.append(llm_reply) # Add this "decision" to history
            force_fetch_url = None # Reset the flag
            # Skip LLM call for this cycle, go straight to tool invocation
        else:
            # === LLM Call (if not manually fetching) ===
            try:
                llm_result = llm.run(messages=current_turn_messages)
                llm_reply = llm_result["replies"][0]
                current_turn_messages.append(llm_reply) # Add LLM response to this turn's context
            except Exception as llm_err:
                print(f"\n--- Error during LLM call in cycle {inner_loop_count} ---")
                print(f"Error: {llm_err}")
                final_reply_for_turn = ChatMessage.from_system("An internal error occurred during LLM communication.")
                break # Exit inner loop

        # === Process LLM Reply (or manual fetch trigger) ===
        if not llm_reply or not llm_reply.tool_calls:
            # LLM responded without tool calls - this is the final response
            final_reply_for_turn = llm_reply if llm_reply else ChatMessage.from_system("No response generated.")
            print("LLM provided final response or no tool calls requested.")
            break # Exit inner loop

        # === Tool Calls Requested (or manually triggered) ===
        print(f"Processing tool calls: {[asdict(tc) for tc in llm_reply.tool_calls]}")
        tool_invoker_input_message = llm_reply
        print("Invoking tools...")
        try:
            tool_result = tool_invoker.run(messages=[tool_invoker_input_message])
            tool_messages = tool_result["tool_messages"] # Successful results
            print(f"Tool results: {[tm.to_dict() for tm in tool_messages]}")
            current_turn_messages.extend(tool_messages) # Add success results

            # --- Check for Manual Fetch Trigger Condition ---
            user_wants_fetch = any(k in original_user_input.lower() for k in ["fetch", "full", "display", "article"])
            search_was_successful = False
            search_tool_message: Optional[ChatMessage] = None

            if tool_messages:
                 # Find the result message corresponding to the brave_web_search call
                 for msg in tool_messages:
                     # Check if it's a TOOL message and contains a ToolCallResult
                     if msg.role == ChatRole.TOOL and msg._content and isinstance(msg._content[0], ToolCallResult):
                         origin = msg._content[0].origin
                         if origin and origin.tool_name == "brave_web_search":
                             search_was_successful = True
                             search_tool_message = msg
                             break # Found the search result

            if user_wants_fetch and search_was_successful and search_tool_message:
                search_result_text = search_tool_message.text or "" # Get text from the correct message
                urls_found = re.findall(r'URL: (https?://[^\s\\]+)', search_result_text)
                if urls_found:
                    url_to_fetch = urls_found[0]
                    # Check if we already tried fetching this URL *in this turn*
                    already_fetched_this_turn = False
                    # Check messages added *after* the user message for this turn
                    for msg in current_turn_messages[turn_start_index_in_messages + 1:]:
                        if msg.role == ChatRole.TOOL and msg._content and isinstance(msg._content[0], ToolCallResult):
                            origin = msg._content[0].origin
                            if origin and origin.tool_name == "fetch" and origin.arguments.get("url") == url_to_fetch:
                                already_fetched_this_turn = True
                                break

                    if not already_fetched_this_turn:
                        force_fetch_url = url_to_fetch # Set flag for next cycle
                        print(f"User wants fetch and search succeeded. Will force fetch for: {force_fetch_url}")
                    else:
                        print(f"Already attempted fetch for {url_to_fetch} this turn, not forcing.")
                else:
                    print("User wants fetch, but no URL found in search results.")
                    # Let LLM decide next step

        except ToolInvocationError as tool_err:
            print(f"\n--- Tool Invocation Error ---")
            print(f"Error: {tool_err}")
            failed_tool_call: Optional[ToolCall] = llm_reply.tool_calls[0] if llm_reply.tool_calls else None
            if failed_tool_call:
                error_message_text = f"Tool '{failed_tool_call.tool_name}' failed with error: {tool_err}"
                tool_error_message = ChatMessage.from_tool(
                    tool_result=error_message_text, origin=failed_tool_call, error=True
                )
                current_turn_messages.append(tool_error_message)
                print(f"Created tool error message: {tool_error_message.to_dict()}")
            else:
                print("Could not determine which tool call failed.")
            # Let the loop continue, LLM will see the error message

        # --- Continue inner loop ---

    # --- After Inner Loop ---
    if final_reply_for_turn:
        print("\nAgent:", final_reply_for_turn.text)
        # Update persistent history
        # Add messages from the start of this turn (user_message) up to the end
        new_history_part = current_turn_messages[len(conversation_history):]
        conversation_history.extend(new_history_part)

    elif inner_loop_count >= MAX_TOOL_CALL_LOOPS:
        print("\nAgent: Reached maximum interaction loops for this turn.")
        conversation_history.append(user_message)
        conversation_history.append(ChatMessage.from_system("Reached maximum interaction loops."))
    else:
        print("\nAgent: Sorry, something went wrong and no final response was generated.")
        conversation_history.append(user_message)