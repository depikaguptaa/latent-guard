"""Demo: hold a flagged answer before the user sees it.

This is the policy Latent describes on its own site (hold, replace with
your text, or hand to a stronger model), shown with fixed scripted
answers because this environment has no LLM key and no Latent key.
Run: python3 demo/hold_demo.py

No network calls are made. The scorer is the local one in latent_guard.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latent_guard.scorer import score_answer  # noqa: E402

CONTEXT = ("Aurora Growth Fund Q3 2026 report. Revenue was $4.2 million, up 12% "
           "from Q2 2026. Priya Sharma is the fund manager.")
SAFE_FALLBACK = "I could not verify that answer against the source report, so I am holding it for review."

ANSWERS = [
    "Revenue was $4.2 million, up 12% from Q2 2026.",
    "Revenue was $9.9 million, up 12% from Q2 2026.",
]


def deliver(answer: str) -> str:
    result = score_answer("Summarize the report.", answer, CONTEXT)
    if result["flagged"]:
        return f"HELD (risk {result['risk']}): {SAFE_FALLBACK}"
    return f"DELIVERED: {answer}"


if __name__ == "__main__":
    for answer in ANSWERS:
        print(deliver(answer))
