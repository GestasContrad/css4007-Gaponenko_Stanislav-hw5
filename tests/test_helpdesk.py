import os
import sys

# project root for sys.path so pytest finds part1, part2, helpdesk, and tools
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import AIMessage, ToolMessage

from part1 import graph_v1, graph_v2
from part2 import build_part2_graph
from helpdesk import run_single_turn, load_user_preferences
from tools import (
    find_instructions,
    check_service_status,
    prepare_ticket,
    set_session_context,
)


class FakeHelpdeskChatModel:
    """Fake tool-capable model used for automated testing without an API key."""

    def __init__(self, mode: str = "auto"):
        self.mode = mode
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        last_msg = messages[-1]

        if self.mode == "infinite_loop":
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "check_service_status",
                        "args": {"service": "wifi"},
                        "id": f"call_loop_{self.calls}",
                    }
                ],
            )

        if isinstance(last_msg, ToolMessage):
            if last_msg.name == "check_service_status":
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "prepare_ticket",
                            "args": {
                                "category": "wifi",
                                "summary": "Wi-Fi connection failed",
                                "steps_tried": ["Enabled Wi-Fi", "Forgot network"],
                                "service_status": "operational",
                            },
                            "id": f"call_ticket_{self.calls}",
                        }
                    ],
                )
            return AIMessage(content="Draft created from tool results.")

        sys_text = messages[0].content if messages else ""
        user_text = last_msg.content.lower()

        if "check whether wi-fi is down" in user_text:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "check_service_status",
                        "args": {"service": "wifi"},
                        "id": f"call_status_{self.calls}",
                    }
                ],
            )

        if "preferred_language = 'Kazakh'" in sys_text:
            return AIMessage(content="Сәлеметсіз бе! Мен қазақ тілінде жауап беремін.")

        return AIMessage(content=f"Echo response for: {last_msg.content}")


def test_part1_routing_and_clarification():
    """Checks case-insensitive routing and the difference between original and updated Part 1 graphs."""
    res_wifi = graph_v1.invoke({"message": "I cannot connect to Wi-Fi."})
    assert res_wifi["category"] == "wifi"
    assert "Campus-Demo" in res_wifi["reply"]

    res_pw = graph_v1.invoke({"message": "I forgot my PASSWORD."})
    assert res_pw["category"] == "password"
    assert "Forgot password" in res_pw["reply"]

    res_other = graph_v1.invoke({"message": "My laptop screen is broken."})
    assert res_other["category"] == "other"

    res_both_v1 = graph_v1.invoke({"message": "My Wi-Fi password does not work."})
    assert res_both_v1["category"] == "wifi"

    res_both_v2 = graph_v2.invoke({"message": "My Wi-Fi password does not work."})
    assert res_both_v2["category"] == "both"
    assert "connecting to Wi-Fi, or recovering your account password" in res_both_v2["reply"]


def test_three_tools_directly(tmp_path, monkeypatch):
    """Checks find_instructions, check_service_status across profiles, and prepare_ticket validation."""
    monkeypatch.setattr("tools.OUTPUTS_DIR", str(tmp_path))
    set_session_context("demo-user-a", "pytest-thread", "normal", lock_profile=False)

    inst = find_instructions("wifi")
    assert inst["instruction_id"] == "GUIDE-WIFI"
    assert len(inst["steps"]) == 4

    st_normal = check_service_status("wifi", status_profile="normal")
    assert st_normal["status"] == "operational"

    st_outage = check_service_status("wifi", status_profile="wifi_outage")
    assert st_outage["status"] == "outage"

    st_unknown = check_service_status("password", status_profile="status_unavailable")
    assert st_unknown["service"] == "account_recovery"
    assert st_unknown["status"] == "unknown"

    bad_ticket = prepare_ticket(category="wifi", summary="Wi-Fi broken", steps_tried=[])
    assert bad_ticket["created"] is False

    good_ticket = prepare_ticket(
        category="wifi",
        summary="Cannot connect to Campus-Demo",
        steps_tried=["Enabled Wi-Fi"],
        service_status="operational",
        user_id="demo-user-a",
        thread_id="pytest-thread",
    )
    assert good_ticket["created"] is True
    assert good_ticket["ticket"]["status"] == "draft"
    assert (tmp_path / os.path.basename(str(good_ticket["file_path"]))).exists()


def test_agent_multi_tool_and_loop_limit_with_fake_model(tmp_path, monkeypatch):
    """Checks multi-tool execution and verifies that MAX_TOOL_LOOPS stops an infinite loop."""
    monkeypatch.setattr("tools.OUTPUTS_DIR", str(tmp_path))
    fake_model = FakeHelpdeskChatModel(mode="auto")
    graph = build_part2_graph(model=fake_model)

    out = graph.invoke(
        {
            "messages": [("user", "Check whether Wi-Fi is down and prepare a ticket.")],
            "user_id": "demo-user-a",
            "thread_id": "pytest-loop",
            "status_profile": "normal",
            "loop_count": 0,
            "loop_limit_reached": False,
            "ticket_files": [],
        }
    )
    assert out["loop_limit_reached"] is False
    assert len(out["ticket_files"]) == 1

    looping_model = FakeHelpdeskChatModel(mode="infinite_loop")
    loop_graph = build_part2_graph(model=looping_model)
    loop_out = loop_graph.invoke(
        {
            "messages": [("user", "Loop forever")],
            "user_id": "demo-user-a",
            "thread_id": "pytest-inf",
            "status_profile": "normal",
            "loop_count": 0,
            "loop_limit_reached": False,
            "ticket_files": [],
        }
    )
    assert loop_out["loop_limit_reached"] is True


def test_memory_persistence_and_user_isolation(tmp_path, monkeypatch):
    """Checks conversation memory across restarts, cross-thread user preferences, and user isolation."""
    monkeypatch.setenv("HELPDESK_STORAGE_DIR", str(tmp_path))
    fake_model = FakeHelpdeskChatModel(mode="auto")

    # 1) demo-user-a saves Kazakh preference in thread-1
    turn1 = run_single_turn(
        user_id="demo-user-a",
        thread_id="thread-1",
        status_profile="normal",
        message="Please remember that I prefer replies in Kazakh. My Wi-Fi fails.",
        model=fake_model,
    )
    assert turn1["user_preference"].get("preferred_language") == "Kazakh"

    # 2) Simulate restart on same thread-1: conversation_state and user_memory must both be present
    turn2 = run_single_turn(
        user_id="demo-user-a",
        thread_id="thread-1",
        status_profile="normal",
        message="Which problem did we discuss?",
        model=fake_model,
    )
    assert "conversation_state" in turn2["sources"]
    assert "user_memory" in turn2["sources"]

    # 3) Same user (demo-user-a), brand new thread-2: user_memory persists without conversation_state
    turn3 = run_single_turn(
        user_id="demo-user-a",
        thread_id="thread-2",
        status_profile="normal",
        message="What language do I prefer?",
        model=fake_model,
    )
    assert "user_memory" in turn3["sources"]
    assert "conversation_state" not in turn3["sources"]
    assert "қазақ" in turn3["final_reply"].lower()

    # 4) Different user (demo-user-b) in thread-3: must not receive demo-user-a's preferences
    turn4 = run_single_turn(
        user_id="demo-user-b",
        thread_id="thread-3",
        status_profile="normal",
        message="Do I have any saved preference?",
        model=fake_model,
    )
    assert turn4["user_preference"] == {}
    assert load_user_preferences("demo-user-b") == {}
    assert "user_memory" not in turn4["sources"]