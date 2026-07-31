import os
from dotenv import load_dotenv
load_dotenv(dotenv_path = os.path.join(os.path.dirname(__file__), '.env'))

from langsmith import Client
import pandas as pd

client = Client()

# ===== TASK COMPLETION (tool runs) =====
tool_runs = client.list_runs(project_name = "hopebot-3-0-phase1-pilot", run_type = "tool")

tool_results = []

for run in tool_runs:
    tool_name = run.name
    output_dict = run.outputs.get("output", {}) if run.outputs else {}
    
    if isinstance(output_dict, dict):
        content = output_dict.get("content", "")
        status = output_dict.get("status", "unknown")
    else:
        content = str(output_dict)
        status = "unknown"
    
    participant_id = run.extra.get("metadata", {}).get("participant_id") if run.extra else None

    # Use LangChain's own status field as primary indicator, with content-based fallback check
    failure_indicators = ["An error occurred", "No relevant", "No psychoeducational content", "No relevant session preparation"]
    success = (status == "success") and not any(content.startswith(indicator) for indicator in failure_indicators)

    tool_results.append({
        "run_id": run.id,
        "participant_id": participant_id,
        "tool_name": tool_name,
        "output": content,
        "status": status,
        "success": success
    })

tool_df = pd.DataFrame(tool_results)
print(tool_df["participant_id"].unique())
print(tool_df)

tool_df.to_csv("phase2_dev_task_completion.csv", index = False)

# ===== LATENCY & TOKEN CONSUMPTION (LLM RUNS) =====
llm_runs = client.list_runs(project_name="hopebot-3-0-dev-testing", run_type="llm")

llm_results = []

for run in llm_runs:
    # Extract participant_id from metadata
    participant_id = run.extra.get("metadata", {}).get("participant_id") if run.extra else None

    latency = (run.end_time - run.start_time).total_seconds() if run.end_time and run.start_time else None
    total_tokens = run.total_tokens if hasattr(run, "total_tokens") else None

    llm_results.append({
        "run_id": run.id,
        "participant_id": participant_id,
        "latency_seconds": latency,
        "total_tokens": total_tokens
    })

llm_df = pd.DataFrame(llm_results)
llm_df.to_csv("phase2_dev_latency_tokens.csv", index = False)

# ===== SUMMARY STATISTICS =====
print(f"Number of tool runs found: {len(list(client.list_runs(project_name='hopebot-3-0-dev-testing', run_type='tool')))}")
# Calculate overall task completion rate per participant
print("Task completion rate by participant:")
print(tool_df.groupby("participant_id")["success"].mean())


print("\nAverage latency and tokens by participant:")
print(llm_df.groupby("participant_id")[["latency_seconds", "total_tokens"]].mean())