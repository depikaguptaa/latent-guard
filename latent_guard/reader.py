"""Read an answer with Latent's real reader, via the runlatent SDK.

This wrapper keeps the SDK's own semantics intact, instead of inventing
a parallel contract:

- `runlatent.check(prompt, answer, context=...)` is the call. It renders
  context as a `Knowledge:` line in the messages and posts messages,
  output, and context to `/score`. This module does not reimplement that.
- A scored result carries `risk`, `verdict` (`pass`, `escalate`, or
  `ood`), and, with `detail=True`, sentence rows with start/end offsets,
  text, and risk, plus probe suite heads. No failure kinds, no reasons.
- A result that was not scored carries `verdict` `unscored` (empty
  answer, nothing to read) or `unknown` (the read did not happen, or came
  back in a shape the SDK cannot read), with `status`, `detail`,
  `resume`, and `retry_after_s` saying why. The caller, not this module,
  decides fail-open vs fail-closed.

Without the SDK installed, or without `LATENT_API_URL` and
`LATENT_API_KEY`, the reader reports itself as not configured and
`read()` returns a not-configured result. It never falls back to the
local regex checker silently: falling back would present local guesses
as the reader's verdict.
"""

from __future__ import annotations

import os
from typing import Any


class LatentReader:
    """Thin wrapper around runlatent.check(). Import is lazy."""

    def __init__(self, api_url: str | None = None, api_key: str | None = None) -> None:
        self.api_url = api_url or os.environ.get("LATENT_API_URL", "")
        self.api_key = api_key or os.environ.get("LATENT_API_KEY", "")
        self._sdk = None
        self._sdk_error: str | None = None
        try:
            import runlatent  # noqa: PLC0415 - optional dependency, lazy

            self._sdk = runlatent
        except ImportError as exc:
            self._sdk_error = str(exc)

    @property
    def configured(self) -> bool:
        return self._sdk is not None and bool(self.api_url) and bool(self.api_key)

    @property
    def config_error(self) -> str | None:
        if self._sdk is None:
            return f"runlatent is not installed: {self._sdk_error}"
        if not self.api_url or not self.api_key:
            return "LATENT_API_URL and LATENT_API_KEY are not both set"
        return None

    def read(
        self,
        prompt: str,
        answer: str,
        context: str | None = None,
        timeout: float = 2.0,
    ) -> dict[str, Any]:
        """Score one answer with the real reader. Never raises on refusal."""
        if not self.configured:
            return {
                "scored": False,
                "verdict": "unknown",
                "risk": None,
                "sentences": None,
                "threshold": None,
                "unscored_reason": "not_configured",
                "detail": self.config_error,
                "status": 0,
                "request_id": None,
            }
        try:
            result = self._sdk.check(
                prompt, answer, context=context, timeout=timeout, detail=True
            )
        except Exception as exc:  # noqa: BLE001 - SDK refusals are data here
            return {
                "scored": False,
                "verdict": "unknown",
                "risk": None,
                "sentences": None,
                "threshold": None,
                "unscored_reason": "read_failed",
                "detail": str(exc),
                "status": 0,
                "request_id": None,
            }
        sentences = None
        if result.sentences is not None:
            sentences = [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "risk": s.risk,
                }
                for s in result.sentences
            ]
        unscored_reason = None
        if not result.scored:
            unscored_reason = "unscored" if result.verdict == "unscored" else "unknown"
            if result.detail:
                unscored_reason = result.detail
        return {
            "scored": bool(result.scored),
            "verdict": result.verdict,
            "risk": result.risk,
            "sentences": sentences,
            "threshold": result.threshold,
            "suite_fired": list(result.suite_fired or ()),
            "unscored_reason": unscored_reason,
            "detail": result.detail,
            "status": result.status,
            "request_id": result.request_id,
        }
