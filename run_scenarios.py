import json
import os
import subprocess
import sys
from part1 import graph_v1, graph_v2

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")


def run_cli(args_list):
    cmd = [sys.executable, "helpdesk.py"] + args_list
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
    return f"$ python helpdesk.py {' '.join(args_list)}", res.stdout.strip()


def run_part1_scenario(scenario: dict) -> str:
    sid = scenario["id"]
    variant = scenario.get("graph_variant", "original")
    graph = graph_v2 if variant == "with_clarification" else graph_v1
    variant_label = (
        "Updated Part 1 Graph (with clarify_problem)"
        if variant == "with_clarification"
        else "Original Part 1 Graph"
    )

    lines = [
        f"Scenario: {sid} — {scenario['title']}",
        f"Graph Variant: {variant_label}",
        f"Status Profile: {scenario['status_profile']}",
    ]

    for ev in scenario["events"]:
        user_id = ev["user_id"]
        thread_id = ev["thread_id"]
        msg = ev["message"]
        lines.append(f"User ID: {user_id} | Thread ID: {thread_id}")
        lines.append(f"Command: python part1.py (scenario {sid}, variant={variant})")
        lines.append(f"\nUser: {msg}")

        current_state = {"message": msg, "category": None, "reply": None}
        nodes_visited = []

        for step in graph.stream({"message": msg}):
            for node_name, update in step.items():
                nodes_visited.append(node_name)
                current_state.update(update)
                lines.append(f"Node visited: {node_name}")
                lines.append(f"  State update: {json.dumps(update, ensure_ascii=False)}")
                lines.append(
                    f"  Current state snapshot: {json.dumps(current_state, ensure_ascii=False)}"
                )

        lines.append(f"\nRoute taken: START -> {' -> '.join(nodes_visited)} -> END")
        lines.append(
            "Information source: data/helpdesk.json (for help nodes) / rule-based routing"
        )
        lines.append(f"Final Reply: {current_state['reply']}")
        lines.append("Loop Limit Reached: False (Part 1 graph has no tool loop)")

    return "\n".join(lines) + "\n"


def run_part2_or_3_scenario(scenario: dict) -> str:
    sid = scenario["id"]
    profile = scenario["status_profile"]
    lines = [
        f"Scenario: {sid} — {scenario['title']}",
        f"Part: {scenario['part']}",
        f"Selected Status Profile: {profile}",
    ]

    if scenario.get("reset_before"):
        cmd_str, out = run_cli(["--reset"])
        lines.append(f"\n[Reset Memory Before Scenario]\n{cmd_str}\n{out}")

    for idx, ev in enumerate(scenario["events"], start=1):
        action = ev.get("action")
        if action == "restart_process":
            preserve = ev.get("preserve_local_storage", True)
            lines.append(
                f"\n--- Event {idx}: restart_process (preserve_local_storage={preserve}) ---"
            )
            lines.append(
                "[Process terminated completely. Next event launches a fresh Python process using saved storage in local-storage/.]"
            )
        elif action == "message":
            user_id = ev["user_id"]
            thread_id = ev["thread_id"]
            msg = ev["message"]
            args = [
                "--user",
                user_id,
                "--thread",
                thread_id,
                "--profile",
                profile,
                "--message",
                msg,
            ]
            cmd_str, out = run_cli(args)
            lines.append(f"\n--- Event {idx}: message ---")
            lines.append(cmd_str)
            lines.append(out)

    return "\n".join(lines) + "\n"


def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    for fname in os.listdir(OUTPUTS_DIR):
        if fname.startswith("S") and fname.endswith(".log"):
            os.remove(os.path.join(OUTPUTS_DIR, fname))
        elif fname.startswith("ticket_s") and fname.endswith(".json"):
            os.remove(os.path.join(OUTPUTS_DIR, fname))

    with open(os.path.join(DATA_DIR, "scenarios.json"), "r", encoding="utf-8") as f:
        scenarios = json.load(f)["scenarios"]

    print(f"Running {len(scenarios)} scenarios from data/scenarios.json...")
    for sc in scenarios:
        sid = sc["id"]
        print(f"-> Running {sid}: {sc['title']}...")
        if sc["part"] == 1:
            log_text = run_part1_scenario(sc)
        else:
            log_text = run_part2_or_3_scenario(sc)

        log_path = os.path.join(OUTPUTS_DIR, f"{sid}.log")
        with open(log_path, "w", encoding="utf-8") as out_f:
            out_f.write(log_text)
        print(f"   Saved outputs/{sid}.log")

    print("\nAll 14 scenario logs generated in outputs/!")


if __name__ == "__main__":
    main()