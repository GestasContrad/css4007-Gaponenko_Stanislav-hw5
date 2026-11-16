import json
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# --- 1. STATE DEFINITION ---
class HelpdeskState(TypedDict):
    message: str
    category: str
    reply: str


with open("data/helpdesk.json", "r", encoding="utf-8") as f:
    helpdesk_data = json.load(f)


# --- 2. ORIGINAL GRAPH (BEFORE CHANGES) ---
def read_request_original(state: HelpdeskState):
    msg = state["message"].lower()
    has_wifi = "wifi" in msg or "wi-fi" in msg
    has_password = "password" in msg

    if has_wifi:
        return {"category": "wifi"}
    elif has_password:
        return {"category": "password"}
    else:
        return {"category": "other"}


def route_request_original(state: HelpdeskState):
    cat = state.get("category")
    if cat == "wifi":
        return "wifi_help"
    elif cat == "password":
        return "password_help"
    else:
        return "other_help"


def wifi_help(state: HelpdeskState):
    steps = helpdesk_data["instructions"]["wifi"]["steps"]
    return {"reply": " ".join(steps)}


def password_help(state: HelpdeskState):
    steps = helpdesk_data["instructions"]["password"]["steps"]
    return {"reply": " ".join(steps)}


def other_help(state: HelpdeskState):
    return {"reply": "Are you having a problem with Wi-Fi or your password?"}


# Orig workflow
workflow_v1 = StateGraph(HelpdeskState)
workflow_v1.add_node("read_request", read_request_original)
workflow_v1.add_node("wifi_help", wifi_help)
workflow_v1.add_node("password_help", password_help)
workflow_v1.add_node("other_help", other_help)

workflow_v1.add_edge(START, "read_request")
workflow_v1.add_conditional_edges(
    "read_request",
    route_request_original,
    ["wifi_help", "password_help", "other_help"]
)
workflow_v1.add_edge("wifi_help", END)
workflow_v1.add_edge("password_help", END)
workflow_v1.add_edge("other_help", END)

graph_v1 = workflow_v1.compile()


# --- 3. UPDATED GRAPH (WITH CLARIFY NODE) ---
def read_request_clarify(state: HelpdeskState):
    msg = state["message"].lower()
    has_wifi = "wifi" in msg or "wi-fi" in msg
    has_password = "password" in msg

    if has_wifi and has_password:
        return {"category": "both"}
    elif has_wifi:
        return {"category": "wifi"}
    elif has_password:
        return {"category": "password"}
    else:
        return {"category": "other"}


def route_request_clarify(state: HelpdeskState):
    cat = state.get("category")
    if cat == "wifi":
        return "wifi_help"
    elif cat == "password":
        return "password_help"
    elif cat == "both":
        return "clarify_problem"
    else:
        return "other_help"


def clarify_problem(state: HelpdeskState):
    return {"reply": "Is the problem connecting to Wi-Fi, or recovering your account password?"}


# Workflow update
workflow_v2 = StateGraph(HelpdeskState)
workflow_v2.add_node("read_request", read_request_clarify)
workflow_v2.add_node("wifi_help", wifi_help)
workflow_v2.add_node("password_help", password_help)
workflow_v2.add_node("other_help", other_help)
workflow_v2.add_node("clarify_problem", clarify_problem)

workflow_v2.add_edge(START, "read_request")
workflow_v2.add_conditional_edges(
    "read_request",
    route_request_clarify,
    ["wifi_help", "password_help", "other_help", "clarify_problem"]
)
workflow_v2.add_edge("wifi_help", END)
workflow_v2.add_edge("password_help", END)
workflow_v2.add_edge("other_help", END)
workflow_v2.add_edge("clarify_problem", END)

graph_v2 = workflow_v2.compile()


# --- 4. RUN TESTS ---
def run_tests():
    test_messages = [
        "I cannot connect to Wi-Fi.",
        "I forgot my PASSWORD.",
        "My laptop screen is broken.",
        "My Wi-Fi password does not work."
    ]

    print("=== TESTING ORIGINAL GRAPH ===")
    for msg in test_messages:
        print(f"\nMessage: {msg}")
        for event in graph_v1.stream({"message": msg}):
            print(f"Node processed: {list(event.keys())[0]}")
            print(f"State update: {event}")

    print("\n\n=== TESTING UPDATED GRAPH (SCENARIO S05) ===")
    msg_conflict = "My Wi-Fi password does not work."
    print(f"Message: {msg_conflict}")
    for event in graph_v2.stream({"message": msg_conflict}):
        print(f"Node processed: {list(event.keys())[0]}")
        print(f"State update: {event}")


if __name__ == "__main__":
    run_tests()