import json
import os
import uuid
from typing import List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

with open(os.path.join(DATA_DIR, "helpdesk.json"), "r", encoding="utf-8") as f:
    HELPDESK_DATA = json.load(f)

with open(os.path.join(DATA_DIR, "service_status.json"), "r", encoding="utf-8") as f:
    STATUS_DATA = json.load(f)

with open(os.path.join(DATA_DIR, "ticket_template.json"), "r", encoding="utf-8") as f:
    TICKET_TEMPLATE = json.load(f)


SESSION_CONTEXT = {
    "user_id": "demo-user-a",
    "thread_id": "default",
    "status_profile": "normal",
    "lock_profile": False,
}


def set_session_context(
    user_id: str = "demo-user-a",
    thread_id: str = "default",
    status_profile: str = "normal",
    lock_profile: bool = True,
):
    SESSION_CONTEXT["user_id"] = user_id
    SESSION_CONTEXT["thread_id"] = thread_id
    SESSION_CONTEXT["status_profile"] = status_profile
    SESSION_CONTEXT["lock_profile"] = lock_profile


def _normalize_topic(raw: str) -> Optional[str]:
    text = (raw or "").strip().lower()
    if "wifi" in text or "wi-fi" in text:
        return "wifi"
    if "password" in text or "recovery" in text or "account" in text:
        return "password"
    return None


def find_instructions(topic: str) -> dict:
    """Look up Wi-Fi or password troubleshooting instructions in the local helpdesk data.

    Args:
        topic: The problem topic to look up ('wifi' or 'password').
    """
    canonical = _normalize_topic(topic)
    if not canonical or canonical not in HELPDESK_DATA["instructions"]:
        return {
            "error": f"No instructions found for topic '{topic}'. Supported topics: wifi, password.",
            "source": "data/helpdesk.json",
        }

    item = HELPDESK_DATA["instructions"][canonical]
    return {
        "topic": canonical,
        "instruction_id": item["instruction_id"],
        "title": item["title"],
        "steps": item["steps"],
        "source": "data/helpdesk.json",
    }


def check_service_status(service: str, status_profile: Optional[str] = None) -> dict:
    """Check whether a university service ('wifi' or 'account_recovery') is operational, in outage, or unknown.

    Args:
        service: Service or topic name ('wifi', 'password', or 'account_recovery').
        status_profile: Optional snapshot profile name for direct testing.
    """
    text = (service or "").strip().lower()
    if text in ("wifi", "wi-fi"):
        service_key = "wifi"
    elif text in ("password", "account_recovery", "recovery", "account"):
        service_key = "account_recovery"
    else:
        return {
            "error": f"Unknown service '{service}'. Use 'wifi' or 'account_recovery'.",
            "source": "data/service_status.json",
        }

    
    if SESSION_CONTEXT.get("lock_profile") or not status_profile:
        profile_name = SESSION_CONTEXT.get("status_profile", "normal")
    else:
        profile_name = status_profile

    profiles = STATUS_DATA.get("profiles", {})
    if profile_name not in profiles:
        return {
            "error": f"Unknown status profile '{profile_name}'.",
            "source": "data/service_status.json",
        }

    service_info = profiles[profile_name].get(service_key, {})
    return {
        "service": service_key,
        "status_profile": profile_name,
        "status": service_info.get("status", "unknown"),
        "message": service_info.get("message", "No message available."),
        "source": "data/service_status.json",
    }


def prepare_ticket(
    category: str,
    summary: str,
    steps_tried: List[str],
    service_status: str = "not_checked",
    reference_ids: Optional[List[str]] = None,
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    status_profile: Optional[str] = None,
) -> dict:
    """Prepare and save a local support ticket draft from the user's problem summary and steps already tried.

    Args:
        category: Problem category ('wifi' or 'password').
        summary: Short description of the user's problem.
        steps_tried: List of troubleshooting steps the user explicitly stated they already tried.
        service_status: Known service status ('operational', 'outage', 'unknown', or 'not_checked').
        reference_ids: Optional list of guide IDs such as ['GUIDE-WIFI'].
    """
    clean_summary = (summary or "").strip()
    clean_steps = [s.strip() for s in (steps_tried or []) if s and s.strip()]

    if not clean_summary or not clean_steps:
        return {
            "error": "Missing required details. Both 'summary' and non-empty 'steps_tried' from the user are required before preparing a ticket.",
            "created": False,
        }

    canonical_cat = _normalize_topic(category) or category.strip().lower()
    active_user = user_id or SESSION_CONTEXT.get("user_id", "demo-user-a")
    active_thread = thread_id or SESSION_CONTEXT.get("thread_id", "default")
    active_profile = (
        SESSION_CONTEXT.get("status_profile", "normal")
        if (SESSION_CONTEXT.get("lock_profile") or not status_profile)
        else status_profile
    )

    if reference_ids is None:
        if canonical_cat == "wifi":
            reference_ids = ["GUIDE-WIFI"]
        elif canonical_cat == "password":
            reference_ids = ["GUIDE-PASSWORD"]
        else:
            reference_ids = []

    ticket_id = f"TCK-{active_thread.upper()}-{uuid.uuid4().hex[:4].upper()}"

    draft = json.loads(json.dumps(TICKET_TEMPLATE["ticket"]))
    draft["ticket_id"] = ticket_id
    draft["status"] = "draft"
    draft["user_id"] = active_user
    draft["thread_id"] = active_thread
    draft["category"] = canonical_cat
    draft["summary"] = clean_summary
    draft["steps_tried"] = clean_steps
    draft["service_status"] = service_status
    draft["status_profile"] = active_profile
    draft["reference_ids"] = reference_ids

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    filename = f"ticket_{active_thread}_{ticket_id}.json"
    rel_path = f"outputs/{filename}"
    full_path = os.path.join(OUTPUTS_DIR, filename)

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump({"ticket": draft}, f, indent=2, ensure_ascii=False)

    return {
        "created": True,
        "delivery": "local draft only",
        "file_path": rel_path,
        "ticket": draft,
    }


TOOLS = [find_instructions, check_service_status, prepare_ticket]
TOOLS_BY_NAME = {fn.__name__: fn for fn in TOOLS}