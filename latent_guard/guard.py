"""Guardrail: act on the reader's verdict before an answer is delivered.

Policies, matching the options Latent's own site describes (hold,
replace with your text, or hand to a stronger model):

- "hold": a flagged answer is replaced with `fallback_text`.
- "replace": same as hold, with the replacement text you supply.
- "reroute": a flagged answer is replaced by the stronger model's
  answer, from the `stronger_model` callback you supply.
- "deliver": never hold a scored answer; the verdict is recorded and the answer goes out.

When the read did not happen (`scored` is False), the guard follows the
fail-open / fail-closed setting you chose, and says which one it
followed in the returned `action`. Fail-open delivers the answer with
action `unscored_delivered`. Fail-closed holds it with action
`unscored_held`. There is no silent third option.

The local regex checker can be passed as `local_scorer` for use as a
pre-filter only. When it is used, the result says
`source: "local"`, the verdict stays the local checker's verdict, and
the answer is never presented as read by Latent. When neither a reader
nor a local scorer is given, the guard returns unscored, as above.
"""

from __future__ import annotations

from typing import Any, Callable

from latent_guard.reader import LatentReader
from latent_guard.scorer import score_answer

POLICIES = ("hold", "replace", "reroute", "deliver")


def guard_answer(
    prompt: str,
    answer: str,
    context: str | None = None,
    *,
    policy: str = "hold",
    fallback_text: str = "I could not verify that answer against the source, so I am holding it for review.",
    stronger_model: Callable[[str, str], str] | None = None,
    fail_closed: bool = False,
    reader: LatentReader | None = None,
    local_scorer: Callable[..., dict[str, Any]] | None = None,
    timeout: float = 2.0,
) -> dict[str, Any]:
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")

    read: dict[str, Any] | None = None
    source = "none"
    if reader is not None:
        read = reader.read(prompt, answer, context, timeout=timeout)
        source = "reader"
    elif local_scorer is not None:
        local = local_scorer(prompt, answer, context or "")
        read = {
            "scored": True,
            "verdict": local["verdict"],
            "risk": local["risk"],
            "sentences": [
                {
                    "start": None,
                    "end": None,
                    "text": s["text"],
                    "risk": s["risk"],
                }
                for s in local["sentences"]
            ],
            "threshold": None,
            "unscored_reason": None,
        }
        source = "local"

    if read is None or not read.get("scored"):
        if fail_closed:
            return {
                "delivered_answer": fallback_text,
                "action": "unscored_held",
                "verdict": (read or {}).get("verdict", "unknown"),
                "risk": (read or {}).get("risk"),
                "sentences": (read or {}).get("sentences"),
                "source": source,
                "fail_mode": "closed",
                "unscored_reason": (read or {}).get("unscored_reason"),
            }
        return {
            "delivered_answer": answer,
            "action": "unscored_delivered",
            "verdict": (read or {}).get("verdict", "unknown"),
            "risk": (read or {}).get("risk"),
            "sentences": (read or {}).get("sentences"),
            "source": source,
            "fail_mode": "open",
            "unscored_reason": (read or {}).get("unscored_reason"),
        }

    flagged = read["verdict"] in ("escalate", "ood")
    if not flagged or policy == "deliver":
        return {
            "delivered_answer": answer,
            "action": "delivered",
            "verdict": read["verdict"],
            "risk": read["risk"],
            "sentences": read["sentences"],
            "source": source,
            "fail_mode": None,
            "unscored_reason": None,
        }
    if policy == "reroute" and stronger_model is not None:
        return {
            "delivered_answer": stronger_model(prompt, answer),
            "action": "rerouted",
            "verdict": read["verdict"],
            "risk": read["risk"],
            "sentences": read["sentences"],
            "source": source,
            "fail_mode": None,
            "unscored_reason": None,
        }
    return {
        "delivered_answer": fallback_text,
        "action": "held" if policy == "hold" else "replaced",
        "verdict": read["verdict"],
        "risk": read["risk"],
        "sentences": read["sentences"],
        "source": source,
        "fail_mode": None,
        "unscored_reason": None,
    }


def guard_stream(
    prompt: str,
    answer_chunks: list[str],
    context: str | None = None,
    *,
    reader: LatentReader | None = None,
    fail_closed: bool = False,
) -> dict[str, Any]:
    """Cut a streamed answer at the first unsupported sentence.

    The reader scores finished answers, so streaming is approximated
    here by scoring the finished answer once, then cutting at the first
    flagged sentence using the reader's offsets. The returned
    `delivered_answer` is the prefix that passed, and `cut_at` names
    the sentence index where delivery stopped, or None if nothing
    was cut. This is a local approximation, not Latent's in-band
    mid-stream cut on its own model's activations.
    """
    if reader is None or not reader.configured:
        return {
            "delivered_answer": "" if fail_closed else "".join(answer_chunks),
            "action": "unscored_delivered" if not fail_closed else "unscored_held",
            "cut_at": None,
            "source": "none",
        }
    # Score the full answer once, then cut at the first flagged sentence.
    full = "".join(answer_chunks)
    read = reader.read(prompt, full, context)
    if not read.get("scored"):
        return {
            "delivered_answer": full if not fail_closed else "",
            "action": "unscored_delivered" if not fail_closed else "unscored_held",
            "cut_at": None,
            "source": "reader",
        }
    sentences = read.get("sentences") or []
    flagged = read["verdict"] in ("escalate", "ood")
    if flagged and not sentences:
        # The reader flagged the answer but gave no sentence rows:
        # there is no safe prefix to deliver, so cut at the start.
        return {
            "delivered_answer": "",
            "action": "cut",
            "cut_at": 0,
            "source": "reader",
        }
    threshold = read.get("threshold")
    for i, sentence in enumerate(sentences):
        risk = sentence.get("risk") or 0.0
        if threshold is not None:
            over = risk >= threshold
        else:
            # No calibrated cut came back: any sentence with risk in a
            # flagged verdict counts as the cut point.
            over = flagged and risk > 0
        if over:
            prefix = full[: sentence.get("start") or 0]
            return {
                "delivered_answer": prefix,
                "action": "cut",
                "cut_at": i,
                "source": "reader",
            }
    return {"delivered_answer": full, "action": "delivered", "cut_at": None, "source": "reader"}
