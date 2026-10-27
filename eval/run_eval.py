"""Run the same cases through the local checker and the real reader.

The suite is all 54 cases: the 30 adversarial cases in
eval/adversarial_cases.json plus the 24 benchmark cases in
benchmark/cases.json (clean maps to pass, hallucinated maps to
escalate). Case ids repeat across the two files, so every row carries
its source suite.

The reader column is never invented. Without LATENT_API_URL and
LATENT_API_KEY (and runlatent installed) every reader result is
reported as not_run, with unscored counts of zero and no precision or
recall for the reader. With a key, the reader is called once per case
through latent_guard.reader.LatentReader, and the comparison is
written to eval/results.json: per-system precision and recall on the
pass/escalate cases, unscored counts, mean latency, and every case
where the two systems disagree.

No-source cases (expected "unscored") are counted separately: a
correct reader answer there is unscored, not pass or escalate.

Usage:
  python eval/run_eval.py
  LATENT_API_URL=... LATENT_API_KEY=... python eval/run_eval.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latent_guard.reader import LatentReader  # noqa: E402
from latent_guard.scorer import score_answer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PROMPT = "Summarize the Aurora Growth Fund Q3 2026 report."


def _load_cases() -> list[dict]:
    """All 54 cases, normalized to one shape.

    Adversarial rows keep their own expected verdicts (pass, escalate,
    unscored) and per-case context. Benchmark rows map clean to pass
    and hallucinated to escalate, share the benchmark file's context,
    and load kind null (clean controls) as kind "clean". Case ids
    repeat across the two files (clean-01 exists in both), so every
    row carries its source suite.
    """
    with open(os.path.join(HERE, "adversarial_cases.json"), encoding="utf-8") as f:
        adv = json.load(f)
    cases = []
    for case in adv["cases"]:
        if case["expected"] not in ("pass", "escalate", "unscored"):
            raise ValueError(f"unknown adversarial expected: {case['expected']!r}")
        cases.append({
            "source": "adversarial",
            "id": case["id"],
            "kind": case["kind"],
            "expected": case["expected"],
            "answer": case["answer"],
            "context": case.get("context", adv["context"]),
            "prompt": case.get("prompt", DEFAULT_PROMPT),
        })
    with open(os.path.join(HERE, "..", "benchmark", "cases.json"), encoding="utf-8") as f:
        bench = json.load(f)
    mapping = {"clean": "pass", "hallucinated": "escalate"}
    for case in bench["cases"]:
        if case["expected"] not in mapping:
            raise ValueError(f"unknown benchmark expected: {case['expected']!r}")
        cases.append({
            "source": "benchmark",
            "id": case["id"],
            "kind": case.get("kind") or "clean",
            "expected": mapping[case["expected"]],
            "answer": case["answer"],
            "context": bench["context"],
            "prompt": DEFAULT_PROMPT,
        })
    return cases


def _pr(rows: list[dict], key: str) -> dict:
    scored = [r for r in rows if r["expected"] in ("pass", "escalate")]
    tp = sum(1 for r in scored if r["expected"] == "escalate" and r[key] == "escalate")
    fn = sum(1 for r in scored if r["expected"] == "escalate" and r[key] != "escalate")
    fp = sum(1 for r in scored if r["expected"] == "pass" and r[key] == "escalate")
    tn = sum(1 for r in scored if r["expected"] == "pass" and r[key] != "escalate")
    return {
        "cases": len(scored),
        "true_positives": tp, "false_negatives": fn,
        "false_positives": fp, "true_negatives": tn,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
    }


def main() -> int:
    cases = _load_cases()
    reader = LatentReader()
    reader_configured = reader.configured

    rows = []
    for case in cases:
        context = case["context"]
        prompt = case["prompt"]
        local = score_answer(prompt, case["answer"], context)
        row = {
            "source": case["source"],
            "id": case["id"], "kind": case["kind"], "expected": case["expected"],
            "answer": case["answer"],
            "local_verdict": local["verdict"], "local_risk": local["risk"],
            "local_latency_ms": local["latency_ms"],
            "reader_verdict": "not_run", "reader_risk": None,
            "reader_latency_ms": None, "reader_unscored_reason": None,
        }
        if reader_configured:
            t0 = time.perf_counter()
            read = reader.read(prompt, case["answer"], context)
            row["reader_latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            row["reader_verdict"] = read["verdict"]
            row["reader_risk"] = read["risk"]
            row["reader_unscored_reason"] = read.get("unscored_reason")
            row["reader_sentences"] = read.get("sentences")
        rows.append(row)

    local_stats = _pr(rows, "local_verdict")
    local_stats["mean_latency_ms"] = round(
        sum(r["local_latency_ms"] for r in rows) / len(rows), 3) if rows else None
    if reader_configured:
        reader_stats = _pr(rows, "reader_verdict")
        reader_stats["mean_latency_ms"] = round(
            sum(r["reader_latency_ms"] for r in rows if r["reader_latency_ms"] is not None) / len(rows), 3) if rows else None
        reader_stats["unscored_count"] = sum(
            1 for r in rows if r["reader_verdict"] in ("unscored", "unknown"))
        reader_stats["unscored_correct"] = sum(
            1 for r in rows if r["expected"] == "unscored" and r["reader_verdict"] in ("unscored", "unknown"))
    else:
        reader_stats = {
            "cases": 0, "true_positives": 0, "false_negatives": 0,
            "false_positives": 0, "true_negatives": 0,
            "precision": None, "recall": None, "mean_latency_ms": None,
            "unscored_count": 0, "unscored_correct": 0,
            "status": "not_run",
            "reason": reader.config_error,
        }

    per_kind: dict[str, dict] = {}
    for kind in sorted({r["kind"] for r in rows}):
        subset = [r for r in rows if r["kind"] == kind]
        per_kind[kind] = {
            "n": len(subset),
            "local_escalated": sum(1 for r in subset if r["local_verdict"] == "escalate"),
            "reader_escalated": sum(1 for r in subset if r["reader_verdict"] == "escalate") if reader_configured else None,
        }

    disagreements = [
        r for r in rows
        if reader_configured and r["expected"] in ("pass", "escalate")
        and r["local_verdict"] != r["reader_verdict"]
    ]
    summary = {
        "cases": len(rows),
        "reader_configured": reader_configured,
        "local": local_stats,
        "reader": reader_stats,
        "per_kind": per_kind,
        "disagreements": disagreements,
        "rows": rows,
    }
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"cases={len(rows)} reader_configured={reader_configured}")
    print(f"local precision={local_stats['precision']} recall={local_stats['recall']}")
    if reader_configured:
        print(f"reader precision={reader_stats['precision']} recall={reader_stats['recall']} "
              f"unscored={reader_stats['unscored_count']}")
    else:
        print(f"reader: not run ({reader.config_error})")
    print(f"disagreements={len(disagreements)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
