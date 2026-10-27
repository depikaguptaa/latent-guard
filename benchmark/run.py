"""Run the fixture benchmark and write benchmark/results.json.

Counts are answer-level (escalated vs passed) plus sentence-level
surfacing for the soft unsupported-claim signal, which sits below the
escalation threshold by design. No model, no network, no key involved:
the answers in cases.json are fixed strings, so the run is fully
reproducible.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latent_guard.scorer import score_answer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    with open(os.path.join(HERE, "cases.json"), encoding="utf-8") as f:
        data = json.load(f)
    context = data["context"]
    rows = []
    for case in data["cases"]:
        result = score_answer("Summarize the Aurora Growth Fund Q3 2026 report.",
                              case["answer"], context)
        rows.append({"id": case["id"], "expected": case["expected"], "kind": case["kind"],
                     "verdict": result["verdict"], "risk": result["risk"],
                     "latency_ms": result["latency_ms"],
                     "sentence_risks": [s["risk"] for s in result["sentences"]],
                     "kinds_found": sorted({r["kind"] for s in result["sentences"] for r in s["reasons"]})})
    tp = sum(1 for r in rows if r["expected"] == "hallucinated" and r["verdict"] == "escalate")
    fn = sum(1 for r in rows if r["expected"] == "hallucinated" and r["verdict"] != "escalate")
    fp = sum(1 for r in rows if r["expected"] == "clean" and r["verdict"] == "escalate")
    tn = sum(1 for r in rows if r["expected"] == "clean" and r["verdict"] != "escalate")
    per_kind = {}
    for kind in ("unsupported_number", "wrong_date", "invented_name", "unsupported_claim"):
        subset = [r for r in rows if r["kind"] == kind]
        per_kind[kind] = {
            "n": len(subset),
            "escalated": sum(1 for r in subset if r["verdict"] == "escalate"),
            "surfaced_at_sentence_level": sum(1 for r in subset if any(x > 0 for x in r["sentence_risks"])),
        }
    summary = {
        "cases": len(rows),
        "true_positives": tp, "false_negatives": fn,
        "false_positives": fp, "true_negatives": tn,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "mean_latency_ms": round(sum(r["latency_ms"] for r in rows) / len(rows), 3),
        "per_kind": per_kind,
        "rows": rows,
    }
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"cases={len(rows)} tp={tp} fn={fn} fp={fp} tn={tn} "
          f"precision={summary['precision']} recall={summary['recall']} "
          f"mean_latency_ms={summary['mean_latency_ms']}")
    print(json.dumps(per_kind, indent=1))


if __name__ == "__main__":
    main()
