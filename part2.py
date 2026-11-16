import json
import os
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from tools import (
    find_instructions,
    check_service_status,
    prepare_ticket,
    set_session_context,
    TOOLS,
    TOOLS_BY_NAME,
)

load_dotenv()

MAX_TOOL_LOOPS = 5

SYSTEM_PROMPT = """You are a university IT helpdesk assistant.
You only assist with:
1. Wi-Fi troubleshooting (network: Campus-Demo, guide: GUIDE-WIFI)
2. Password / account recovery guidance (guide: GUIDE-PASSWORD)
3. Service status checks (services: 'wifi' or 'account_recovery')
4. Preparing local support ticket drafts

Strict rules:
- Use `find_instructions` to get troubleshooting steps from the local guide.
- Use `check_service_status` when asked if a service is down or before conditional ticket creation.
- If the user says "If it is operating, prepare a ticket":
  * Only call `prepare_ticket` when `check_service_status` returns status == 'operational'.
  * If status == 'outage', DO NOT prepare a ticket. Explain that there is an outage, no ticket was prepared because the condition was not met, and note that no repair time is given in the data.
  * If status == 'unknown', DO NOT prepare a ticket automatically. State that the status source is unavailable so you cannot confirm it is operating, and ask if the user still wants a ticket draft noting that status could not be checked.
- Never invent troubleshooting steps that the user did not state they tried.
- If a user asks to prepare a ticket without stating both their problem and the steps they already tried, DO NOT call `prepare_ticket`; ask the user for the missing problem description and steps tried first.
- Tickets are saved as local drafts only; never claim a ticket was sent to a real team. If a ticket draft is created, mention its local file_path in your reply.
"""


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    user_id: str
    thread_id: str
    status_profile: str
    loop_count: int
    loop_limit_reached: bool
    ticket_files: List[str]


def build_part2_graph(model=None):
    if model is None:
        model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        model = ChatOpenAI(model=model_name, temperature=0)

    model_with_tools = model.bind_tools(TOOLS)

    def call_model(state: AgentState):
        current_loops = state.get("loop_count", 0) + 1
        msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = model_with_tools.invoke(msgs)
        return {
            "messages": [response],
            "loop_count": current_loops,
        }

    def route_after_model(state: AgentState):
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", None)
        if not tool_calls:
            return "end"
        if state.get("loop_count", 0) >= MAX_TOOL_LOOPS:
            return "limit_reached"
        return "tools"

    def execute_tools(state: AgentState):
        last_msg = state["messages"][-1]
        tool_messages = []
        created_files = list(state.get("ticket_files", []))

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

            if isinstance(result, dict) and result.get("file_path"):
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
        }

    def handle_limit(state: AgentState):
        return {
            "messages": [
                AIMessage(
                    content="Tool loop limit reached. Stopping further tool calls."
                )
            ],
            "loop_limit_reached": True,
        }

    workflow = StateGraph(AgentState)
    workflow.add_node("call_model", call_model)
    workflow.add_node("tools", execute_tools)
    workflow.add_node("handle_limit", handle_limit)

    workflow.add_edge(START, "call_model")
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

    return workflow.compile()


def run_direct_tool_demos():
    print("=== 1. DIRECT TOOL CALLS (WITHOUT AGENT) ===")
    set_session_context("demo-user-a", "direct-test", "normal", lock_profile=False)

    res1 = find_instructions("wifi")
    print("\n[Direct call] find_instructions('wifi'):")
    print(json.dumps(res1, indent=2))

    res2 = check_service_status("wifi", status_profile="normal")
    print("\n[Direct call] check_service_status('wifi', status_profile='normal'):")
    print(json.dumps(res2, indent=2))

    res3 = prepare_ticket(
        category="wifi",
        summary="Cannot connect to Campus-Demo Wi-Fi",
        steps_tried=["Enabled Wi-Fi", "Forgot the saved network"],
        service_status="operational",
        user_id="demo-user-a",
        thread_id="direct-test",
        status_profile="normal",
    )
    print("\n[Direct call] prepare_ticket(...):")
    print(json.dumps(res3, indent=2))


def run_agent_demo(graph, scenario_label: str, thread_id: str, profile: str, user_messages: List[str]):
    print(f"\n=== {scenario_label} (status_profile='{profile}') ===")
    history: List[AnyMessage] = []

    for msg_text in user_messages:
        print(f"\nUser: {msg_text}")
        history.append(HumanMessage(content=msg_text))
        state_in = {
            "messages": history,
            "user_id": "demo-user-a",
            "thread_id": thread_id,
            "status_profile": profile,
            "loop_count": 0,
            "loop_limit_reached": False,
            "ticket_files": [],
        }

        for event in graph.stream(state_in):
            for node_name, update in event.items():
                print(f"Node visited: {node_name}")
                for m in update.get("messages", []):
                    if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                        for tc in m.tool_calls:
                            print(f"  -> Model requested tool: {tc['name']}({tc['args']})")
                    elif isinstance(m, ToolMessage):
                        print(f"  -> Tool result ({m.name}): {m.content}")
                    elif isinstance(m, AIMessage):
                        print(f"Helpdesk: {m.content}")
                if update.get("ticket_files"):
                    print(f"  -> Ticket files: {update['ticket_files']}")


        final_state = graph.invoke(state_in) if False else None

        for event_Msgs in [u.get("messages", []) for u in [ event_val for event_val in [] ]]:
            pass


def run_part2_demos():
    run_direct_tool_demos()
    graph = build_part2_graph()

    # Demo S06
    print("\n\n=== 2. AGENT MULTI-TOOL DEMOS ACROSS SNAPSHOTS ===")
    for label, tid, profile, prompts in [
        (
            "S06 Demo - Normal Status",
            "s06",
            "normal",
            [
                "Show me the Wi-Fi troubleshooting steps.",
                "I enabled Wi-Fi and forgot the saved network, but Campus-Demo still will not connect. Check whether Wi-Fi is down. If it is operating, prepare a ticket about this problem.",
            ],
        ),
        (
            "S07 Demo - Wi-Fi Outage",
            "s07",
            "wifi_outage",
            [
                "I enabled Wi-Fi and forgot the saved network, but Campus-Demo still will not connect. Check whether Wi-Fi is down. If it is operating, prepare a ticket about this problem."
            ],
        ),
        (
            "S08 Demo - Status Unavailable",
            "s08",
            "status_unavailable",
            [
                "Check whether Wi-Fi is down. If it is operating, prepare a ticket. I already enabled Wi-Fi and reconnected, with no success."
            ],
        ),
    ]:
        print(f"\n--- {label} (status_profile={profile}) ---")
        messages: List[AnyMessage] = []
        for prompt in prompts:
            print(f"\nUser: {prompt}")
            messages.append(HumanMessage(content=prompt))
            state_in = {
                "messages": messages,
                "user_id": "demo-user-a",
                "thread_id": tid,
                "status_profile": profile,
                "loop_count": 0,
                "loop_limit_reached": False,
                "ticket_files": [],
            }
            for step in graph.stream(state_in):
                for node_name, update in step.items():
                    print(f"Node visited: {node_name}")
                    for m in update.get("messages", []):
                        messages.append(m)
                        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                            for tc in m.tool_calls:
                                print(f"  [Tool Request] {tc['name']}({tc['args']})")
                        elif isinstance(m, ToolMessage):
                            print(f"  [Tool Result - {m.name}] {m.content}")
                        elif isinstance(m, AIMessage):
                            print(f"Helpdesk: {m.content}")


if __name__ == "__main__":
    run_part2_demos()