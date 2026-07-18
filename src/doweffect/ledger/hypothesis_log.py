import json
import os
from datetime import datetime

def log_hypothesis_result(hypothesis_id: str,
                          description: str,
                          test_name: str,
                          p_value_raw: float,
                          p_value_adj: float,
                          effect_size: float,
                          verdict: str,
                          ledger_path: str = "RESEARCH_LEDGER.md"):
    """
    Appends the result of a hypothesis test to the central Research Ledger.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    entry = (
        f"\n### {hypothesis_id} - Logged at {timestamp}\n"
        f"- **Description:** {description}\n"
        f"- **Test Method:** {test_name}\n"
        f"- **Raw p-value:** {p_value_raw:.4e}\n"
        f"- **Adjusted p-value:** {p_value_adj:.4e}\n"
        f"- **Effect Size:** {effect_size:.4f}\n"
        f"- **Verdict:** **{verdict}**\n"
        "---\n"
    )
    
    # Check if file exists, if not create it
    if not os.path.exists(ledger_path):
        with open(ledger_path, "w") as f:
            f.write("# Research Ledger\n\n")
            
    with open(ledger_path, "a") as f:
        f.write(entry)
