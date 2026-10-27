"""Review queue: every scored answer is stored, flagged ones wait for a
human decision, and each decision becomes one labeled example that can
be exported for fine-tuning or preference data. Storage is one JSON
file so the whole thing runs with zero dependencies.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

DECISIONS = {"confirmed": "hallucinated", "released": "clean"}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _scrub_sentences(sentences) -> list:
    """Hash-only form of sentence rows: no answer text survives.

    In hash mode the prompt and answer fields are hashes, but the
    sentence rows carry the answer's own text and the reasons carry
    its numbers and names. Storing them raw would leak the answer
    through the queue file, so only offsets, risk, and failure kinds
    are kept, with the sentence text itself hashed.
    """
    out = []
    for s in sentences or []:
        if not isinstance(s, dict):
            continue
        out.append({
            "start": s.get("start"),
            "end": s.get("end"),
            "risk": s.get("risk"),
            "text_sha256": _sha(s.get("text") or ""),
            "reasons": [
                {"kind": r.get("kind")}
                for r in (s.get("reasons") or [])
                if isinstance(r, dict) and r.get("kind")
            ],
        })
    return out


class ReviewQueue:
    def __init__(self, path: str, store_text: bool = True):
        """store_text=False keeps hashes of prompt/answer, never the text.

        Latent's own records are hash-only. Use hash mode for anything
        beyond a local demo, so reviewed examples cannot leak the
        original prompt or answer out of the queue file.
        """
        self.path = path
        self.store_text = store_text
        self.entries: dict = {}
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                text = f.read().strip()
            rows = json.loads(text) if text else []
            for row in rows:
                self.entries[row["request_id"]] = row

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(list(self.entries.values()), f, indent=1)
        os.replace(tmp, self.path)

    def add(self, result: dict, prompt: str, answer: str) -> dict:
        if result.get("scored") is False:
            # The queue stores scored answers only: an unscored result
            # (unknown verdict, no read happened) is not a verdict to
            # review, and its request_id may be None. Scorer-shaped
            # results carry no "scored" key and are unaffected.
            raise ValueError("cannot queue a result that was not scored")
        entry = {
            "request_id": result["request_id"],
            "prompt": prompt if self.store_text else _sha(prompt),
            "answer": answer if self.store_text else _sha(answer),
            "prompt_sha256": _sha(prompt),
            "answer_sha256": _sha(answer),
            "verdict": result["verdict"],
            "risk": result["risk"],
            "sentences": result["sentences"] if self.store_text else _scrub_sentences(result["sentences"]),
            "status": "pending_review" if result.get(
                "flagged", result["verdict"] in ("escalate", "ood")) else "delivered",
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
            if e["status"] == "reviewed" and e.get("decision") in DECISIONS:
                # Sentences may be reader-shaped (start/end/text/risk,
                # no reasons) or None: only scorer-shaped rows carry kinds.
                kinds = {
                    r.get("kind")
                    for s in (e.get("sentences") or [])
                    if isinstance(s, dict)
                    for r in (s.get("reasons") or [])
                    if isinstance(r, dict) and r.get("kind")
                }
                out.append({
                    "prompt": e["prompt"],
                    "answer": e["answer"],
                    "label": DECISIONS[e["decision"]],
                    "failure_kinds": sorted(kinds),
                })
        return out

    def metrics(self) -> dict:
        total = len(self.entries)
        flagged = sum(1 for e in self.entries.values() if e["verdict"] in ("escalate", "ood"))
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
