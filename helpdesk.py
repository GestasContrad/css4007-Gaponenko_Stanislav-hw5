import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from typing import Annotated, List, Optional, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from tools import TOOLS, TOOLS_BY_NAME, set_session_context

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

MAX_TOOL_LOOPS = 5
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_storage_dir() -> str:
    storage_rel = os.getenv("HELPDESK_STORAGE_DIR", "local-storage")
    if os.path.isabs(storage_rel):
        return storage_rel
    return os.path.join(BASE_DIR, storage_rel)


def get_checkpoint_db_path() -> str:
    storage_dir = get_storage_dir()
    os.makedirs(storage_dir, exist_ok=True)
    return os.path.join(storage_dir, "checkpoints.sqlite")


def get_user_prefs_path() -> str:
    storage_dir = get_storage_dir()
    os.makedirs(storage_dir, exist_ok=True)
    return os.path.join(storage_dir, "user_preferences.json")


def reset_storage():
    storage_dir = get_storage_dir()
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)
    os.makedirs(storage_dir, exist_ok=True)


def load_user_preferences(user_id: str) -> dict:
    prefs_path = get_user_prefs_path()
    if not os.path.exists(prefs_path):
        return {}
    try:
        with open(prefs_path, "r", encoding="utf-8") as f:
            all_prefs = json.load(f)
        return dict(all_prefs.get(user_id, {}))
    except Exception:
        return {}


def save_user_preference(user_id: str, key: str, value: str) -> dict:
    prefs_path = get_user_prefs_path()
    all_prefs = {}
    if os.path.exists(prefs_path):
        try:
            with open(prefs_path, "r", encoding="utf-8") as f:
                all_prefs = json.load(f)
        except Exception:
            all_prefs = {}

    user_entry = dict(all_prefs.get(user_id, {}))
    user_entry[key] = value
    all_prefs[user_id] = user_entry

    with open(prefs_path, "w", encoding="utf-8") as f:
        json.dump(all_prefs, f, indent=2, ensure_ascii=False)

    return user_entry


def extract_language_preference(text: str) -> Optional[str]:
    patterns = [
        r"prefer\s+replies\s+in\s+([A-Za-z]+)",
        r"reply\s+in\s+([A-Za-z]+)",
        r"respond\s+in\s+([A-Za-z]+)",
        r"language\s+preference\s+is\s+([A-Za-z]+)",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return match.group(1).strip().capitalize()
    return None


class HelpdeskAgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    user_id: str
    thread_id: str
    status_profile: str
    user_preference: dict
    memory_sources: List[str]
    loop_count: int
    loop_limit_reached: bool
    ticket_files: List[str]


def build_system_prompt(user_id: str, user_pref: dict) -> str:
    pref_lang = user_pref.get("preferred_language")
    if pref_lang:
        lang_instruction = (
            f"USER MEMORY RECORD FOR '{user_id}': preferred_language = '{pref_lang}'. "
            f"You MUST reply in {pref_lang}. If the user asks which language they requested, "
            f"explicitly mention '{pref_lang}' in your answer."
        )
    else:
        lang_instruction = (
            f"USER MEMORY RECORD FOR '{user_id}': No saved language preference. "
            f"If the user asks whether they have shared a preferred language or previous problem "
            f"and none exists in this conversation history, clearly state that no preferred language "
            f"or previous problem has been recorded for user '{user_id}'."
        )

    return f"""You are a university IT helpdesk assistant.
{lang_instruction}

Scope of help:
1. Wi-Fi troubleshooting (network: Campus-Demo, guide: GUIDE-WIFI)
2. Password / account recovery guidance (guide: GUIDE-PASSWORD)
3. Service status lookup (services: 'wifi' or 'account_recovery')
4. Preparing local support ticket drafts

Strict rules:
- Use `find_instructions` when the user asks for troubleshooting steps.
- Use `check_service_status` when asked whether a service is down or before conditional ticket creation.
- If the user says "If it is operating, prepare a ticket":
  * Call `prepare_ticket` ONLY when `check_service_status` returns status == 'operational'.
  * If status == 'outage', DO NOT prepare a ticket. Explain that there is an outage, no ticket was prepared because the condition was not met, and state that no repair time is provided in the data.
  * If status == 'unknown', DO NOT prepare a ticket automatically. State that the status source is unavailable so you cannot confirm it is operating, and ask if the user wants a ticket draft noting that status could not be checked.
- Never invent troubleshooting steps that the user did not state they tried.
- If a user asks to prepare a ticket without stating both their problem and the steps they already tried (either in the current message or earlier in this conversation), DO NOT call `prepare_ticket`; ask the user for the missing problem description and steps tried.
- Tickets are saved as local drafts only; never claim a ticket was sent externally. Always include the local `file_path` when a draft is created.
"""


def create_helpdesk_graph(checkpointer=None, model=None):
    if model is None:
        model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        model = ChatOpenAI(model=model_name, temperature=0)

    model_with_tools = model.bind_tools(TOOLS)

    def sync_user_memory(state: HelpdeskAgentState):
        user_id = state["user_id"]
        prefs = load_user_preferences(user_id)
        sources = list(state.get("memory_sources", []))

        if len(state.get("messages", [])) > 1 and "conversation_state" not in sources:
            sources.append("conversation_state")

        if prefs and "user_memory" not in sources:
            sources.append("user_memory")

        # Check the latest human message for a new language preference
        messages = state.get("messages", [])
        if messages and isinstance(messages[-1], HumanMessage):
            detected_lang = extract_language_preference(messages[-1].content)
            if detected_lang:
                prefs = save_user_preference(user_id, "preferred_language", detected_lang)
                if "user_memory" not in sources:
                    sources.append("user_memory")

        return {
            "user_preference": prefs,
            "memory_sources": sources,
        }

    def call_model(state: HelpdeskAgentState):
        current_loops = state.get("loop_count", 0) + 1
        sys_prompt = build_system_prompt(
            state["user_id"], state.get("user_preference", {})
        )
        msgs = [SystemMessage(content=sys_prompt)] + state["messages"]
        response = model_with_tools.invoke(msgs)
        return {
            "messages": [response],
            "loop_count": current_loops,
        }

    def route_after_model(state: HelpdeskAgentState):
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", None)
        if not tool_calls:
            return "end"
        if state.get("loop_count", 0) >= MAX_TOOL_LOOPS:
            return "limit_reached"
        return "tools"

    def execute_tools(state: HelpdeskAgentState):
        last_msg = state["messages"][-1]
        tool_messages = []
        created_files = list(state.get("ticket_files", []))
        sources = list(state.get("memory_sources", []))

        set_session_context(
            user_id=state.get("user_id", "demo-user-a"),
            thread_id=state.get("thread_id", "default"),
            status_profile=state.get("status_profile", "normal"),
            lock_profile=True,
        )

        for tc in last_msg.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_fn = TOOLS_BY_NAME.get(tool_name)

            if tool_fn is None:
                result = {"error": f"Tool '{tool_name}' not found."}
            else:
                result = tool_fn(**tool_args)

            if isinstance(result, dict):
                src = result.get("source")
                if src and src not in sources:
                    sources.append(src)
                if result.get("file_path"):
                    created_files.append(result["file_path"])

            tool_messages.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=tc["id"],
                    name=tool_name,
                )
            )

        return {
            "messages": tool_messages,
            "ticket_files": created_files,
            "memory_sources": sources,
        }

    def handle_limit(state: HelpdeskAgentState):
        return {
            "messages": [
                AIMessage(
                    content="Tool loop limit reached. Stopping further tool calls."
                )
            ],
            "loop_limit_reached": True,
        }

    workflow = StateGraph(HelpdeskAgentState)
    workflow.add_node("user_memory", sync_user_memory)
    workflow.add_node("call_model", call_model)
    workflow.add_node("tools", execute_tools)
    workflow.add_node("handle_limit", handle_limit)

    workflow.add_edge(START, "user_memory")
    workflow.add_edge("user_memory", "call_model")
    workflow.add_conditional_edges(
        "call_model",
        route_after_model,
        {
            "tools": "tools",
            "limit_reached": "handle_limit",
            "end": END,
        },
    )
    workflow.add_edge("tools", "call_model")
    workflow.add_edge("handle_limit", END)

    return workflow.compile(checkpointer=checkpointer)


def run_single_turn(
    user_id: str,
    thread_id: str,
    status_profile: str,
    message: str,
    model=None,
) -> dict:
    db_path = get_checkpoint_db_path()
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        checkpointer = SqliteSaver(conn)
        config = {"configurable": {"thread_id": thread_id}}

        # Check if prior messages already exist in checkpointed thread
        existing_tuple = checkpointer.get_tuple(config)
        had_prior_messages = False
        if existing_tuple and existing_tuple.checkpoint:
            channel_vals = existing_tuple.checkpoint.get("channel_values", {})
            if channel_vals.get("messages"):
                had_prior_messages = True

        graph = create_helpdesk_graph(checkpointer=checkpointer, model=model)

        initial_sources = ["conversation_state"] if had_prior_messages else []
        state_input = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
            "thread_id": thread_id,
            "status_profile": status_profile,
            "memory_sources": initial_sources,
            "loop_count": 0,
            "loop_limit_reached": False,
            "ticket_files": [],
        }

        nodes_visited = []
        tool_requests = []
        tool_results = []
        final_reply = ""
        user_pref = {}
        sources = list(initial_sources)
        ticket_files = []
        limit_reached = False

        print(
            f"[Session] user_id={user_id} | thread_id={thread_id} | status_profile={status_profile}"
        )
        print(f"User: {message}")

        for event in graph.stream(state_input, config=config):
            for node_name, update in event.items():
                nodes_visited.append(node_name)
                print(f"Node visited: {node_name}")

                if "user_preference" in update:
                    user_pref = update["user_preference"]
                    print(f"  [User Memory] preference={user_pref}")

                if "memory_sources" in update:
                    sources = update["memory_sources"]

                if "ticket_files" in update and update["ticket_files"]:
                    ticket_files = update["ticket_files"]

                if update.get("loop_limit_reached"):
                    limit_reached = True

                for m in update.get("messages", []):
                    if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                        for tc in m.tool_calls:
                            req = {"name": tc["name"], "args": tc["args"]}
                            tool_requests.append(req)
                            print(f"  [Tool Request] {tc['name']}({tc['args']})")
                    elif isinstance(m, ToolMessage):
                        res = {"name": m.name, "content": m.content}
                        tool_results.append(res)
                        print(f"  [Tool Result - {m.name}] {m.content}")
                    elif isinstance(m, AIMessage):
                        final_reply = m.content
                        print(f"Helpdesk: {final_reply}")

        print(f"  [Sources Used] {sources}")
        if ticket_files:
            print(f"  [Ticket Drafts] {ticket_files}")
        print(f"  [Loop Limit Reached] {limit_reached}")

        return {
            "user_id": user_id,
            "thread_id": thread_id,
            "status_profile": status_profile,
            "message": message,
            "nodes_visited": nodes_visited,
            "user_preference": user_pref,
            "sources": sources,
            "tool_requests": tool_requests,
            "tool_results": tool_results,
            "final_reply": final_reply,
            "ticket_files": ticket_files,
            "loop_limit_reached": limit_reached,
        }
    finally:
        conn.close()


def run_part3_demos():
    print("=== PART 3 MEMORY DEMONSTRATIONS (VIA SUBPROCESSES) ===")
    reset_storage()

    def call_cli(args_list: List[str]):
        cmd = [sys.executable, "helpdesk.py"] + args_list
        print(f"\n$ {' '.join(cmd)}")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        res = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        print(res.stdout.strip())
        if res.stderr:
            print(res.stderr.strip())

    print("\n--- 1. Refer to an earlier message (S10) ---")
    call_cli([
        "--user", "demo-user-a",
        "--thread", "s10",
        "--profile", "normal",
        "--message", "I cannot connect to Wi-Fi. I enabled it and forgot the saved network, but reconnecting failed."
    ])
    call_cli([
        "--user", "demo-user-a",
        "--thread", "s10",
        "--profile", "normal",
        "--message", "Which steps did I already try?"
    ])

    print("\n--- 2. Close, restart process, and resume same conversation (S11) ---")
    call_cli(["--reset"])
    call_cli([
        "--user", "demo-user-a",
        "--thread", "restart-a",
        "--profile", "normal",
        "--message", "Please remember that I prefer replies in Kazakh. My issue is Wi-Fi. I enabled it and tried reconnecting, but it still fails."
    ])
    print("[Process exited. Launching new process against saved SQLite & JSON storage...]")
    call_cli([
        "--user", "demo-user-a",
        "--thread", "restart-a",
        "--profile", "normal",
        "--message", "Which problem and steps did we discuss?"
    ])

    print("\n--- 3. Same user, new conversation thread (S12) ---")
    call_cli([
        "--user", "demo-user-a",
        "--thread", "preference-a2",
        "--profile", "normal",
        "--message", "Which language did I ask you to use? What can you help with?"
    ])

    print("\n--- 4. Different user isolation (S13) ---")
    call_cli([
        "--user", "demo-user-b",
        "--thread", "isolation-b",
        "--profile", "normal",
        "--message", "Have I told you a preferred language or a previous problem?"
    ])


def main():
    parser = argparse.ArgumentParser(description="University IT Helpdesk CLI")
    parser.add_argument("--user", default="demo-user-a", help="User identifier")
    parser.add_argument("--thread", default="default-thread", help="Conversation thread identifier")
    parser.add_argument("--profile", default="normal", help="Service status snapshot profile")
    parser.add_argument("--message", help="Single message to send to the helpdesk")
    parser.add_argument("--reset", action="store_true", help="Clear local conversation and user memory")
    parser.add_argument("--demo-part3", action="store_true", help="Run Part 3 memory demonstrations")

    args = parser.parse_args()

    if args.reset:
        reset_storage()
        print("Local memory storage cleared.")
        if not args.message and not args.demo_part3:
            return

    if args.demo_part3:
        run_part3_demos()
        return

    if args.message:
        run_single_turn(
            user_id=args.user,
            thread_id=args.thread,
            status_profile=args.profile,
            message=args.message,
        )
        return

    print(
        f"Interactive Helpdesk started (user={args.user}, thread={args.thread}, profile={args.profile}). Type 'exit' to quit."
    )
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except EOFError:
            break
        if not user_input or user_input.lower() in ("exit", "quit"):
            break
        run_single_turn(
            user_id=args.user,
            thread_id=args.thread,
            status_profile=args.profile,
            message=user_input,
        )


if __name__ == "__main__":
    main()