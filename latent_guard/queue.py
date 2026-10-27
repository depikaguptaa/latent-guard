"""Review queue: every scored answer is stored, flagged ones wait for a
human decision, and each decision becomes one labeled example that can
be exported for fine-tuning or preference data. Storage is one JSON
file so the whole thing runs with zero dependencies.
"""

from __future__ import annotations

import json
import os
import time

DECISIONS = {"confirmed": "hallucinated", "released": "clean"}


class ReviewQueue:
    def __init__(self, path: str):
        self.path = path
        self.entries: dict = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for row in json.load(f):
                    self.entries[row["request_id"]] = row

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(list(self.entries.values()), f, indent=1)
        os.replace(tmp, self.path)

    def add(self, result: dict, prompt: str, answer: str) -> dict:
        entry = {
            "request_id": result["request_id"],
            "prompt": prompt,
            "answer": answer,
            "verdict": result["verdict"],
            "risk": result["risk"],
            "sentences": result["sentences"],
            "status": "pending_review" if result["flagged"] else "delivered",
            "decision": None,
            "note": None,
            "created_at": time.time(),
        }
        self.entries[entry["request_id"]] = entry
        self._save()
        return entry

    def review(self, request_id: str, decision: str, note: str = "") -> dict:
        if decision not in DECISIONS:
            raise ValueError("decision must be confirmed or released")
        entry = self.entries[request_id]
        entry["status"] = "reviewed"
        entry["decision"] = decision
        entry["note"] = note
        self._save()
        return entry

    def pending(self) -> list:
        return [e for e in self.entries.values() if e["status"] == "pending_review"]

    def labeled_examples(self) -> list:
        out = []
        for e in self.entries.values():
            if e["status"] == "reviewed":
                out.append({
                    "prompt": e["prompt"],
                    "answer": e["answer"],
                    "label": DECISIONS[e["decision"]],
                    "failure_kinds": sorted({r["kind"] for s in e["sentences"] for r in s["reasons"]}),
                })
        return out

    def metrics(self) -> dict:
        total = len(self.entries)
        flagged = sum(1 for e in self.entries.values() if e["verdict"] == "escalate")
        reviewed = sum(1 for e in self.entries.values() if e["status"] == "reviewed")
        confirmed = sum(1 for e in self.entries.values() if e["decision"] == "confirmed")
        return {
            "total_scored": total,
            "flagged": flagged,
            "flag_rate": round(flagged / total, 4) if total else 0.0,
            "reviewed": reviewed,
            "confirmed_hallucinated": confirmed,
            "precision_on_reviewed": round(confirmed / reviewed, 4) if reviewed else None,
        }
