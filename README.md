# HW5 — Build a university IT helpdesk with LangGraph

## My Implementation - Setup and Run Instructions

**Tested Python Version:** `Python 3.14.0`

### 1. Install dependencies

Create and activate a virtual environment, then install the pinned packages:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and set your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
MODEL_NAME=gpt-4o-mini
HELPDESK_STORAGE_DIR=local-storage
LANGSMITH_TRACING=false
```

### 3. Exact commands to run the program and demonstrations

- **Run Part 1 (keyword routing without an LLM, original vs. updated graph):**
  ```powershell
  python part1.py
  ```
- **Run Part 2 (direct tool calls and multi-tool agent across status snapshots):**
  ```powershell
  python part2.py
  ```
- **Run Part 3 (conversation & user memory demos across separate subprocesses):**
  ```powershell
  python helpdesk.py --demo-part3
  ```
- **Send a single message via the CLI:**
  ```powershell
  python helpdesk.py --user demo-user-a --thread my-thread --profile normal --message "I cannot connect to Wi-Fi."
  ```
- **Start interactive chat mode in the terminal:**
  ```powershell
  python helpdesk.py --user demo-user-a --thread interactive-1 --profile normal
  ```
- **Replay all 14 supplied scenarios (`S01`–`S14`) and generate logs in `outputs/`:**
  ```powershell
  python run_scenarios.py
  ```

### 4. Where memory files are stored and how to reset them

- **Conversation state (LangGraph SQLite checkpointer):** `local-storage/checkpoints.sqlite`
- **User preferences (per-user JSON store):** `local-storage/user_preferences.json`
- **How to clear all saved memory and start fresh:**
  ```powershell
  python helpdesk.py --reset
  ```
  (Alternatively, delete the `local-storage/` directory manually.)

### 5. Run automated tests

The test suite in `tests/test_helpdesk.py` runs completely offline using `FakeHelpdeskChatModel` (no API key required):

```powershell
pytest -v
```


