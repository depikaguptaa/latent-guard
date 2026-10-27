# Latent proof of work: latent_guard

Target: Latent (runlatent.ai), YC Fall 2026, hallucination detection for
AI apps. Founder: Vedant Gaur. Verified contact: founders@runlatent.ai,
from their own company page (runlatent.ai/company, read 2026-10-25).
Booking link on the same page: latent.cal.com/vedant.

## Why Latent fits

* Eval and reliability tooling is exactly her work: LangChain
  multi-agent orchestration, RAG, LLM observability, FastAPI, AWS at
  Outfit Formulas; LangGraph and MCP/FastMCP at Pine Labs.
* Latent's own pitch names the failure kinds this build checks:
  unsupported numbers, wrong dates, invented names, with a review
  queue, Prometheus and OpenTelemetry hooks, and labeled answers kept
  for fine-tuning. This build mirrors that shape, offline.
* No public job posting was found for this Fall 2026 Latent. YC's
  directory page for the name Latent belongs to an older, different
  company (W23, medical language models, 50 people, San Francisco
  onsite). Do not apply through that page. This is a founder play.

## What was built

`latent_guard`: a zero dependency Python package.

* `scorer.py`: deterministic sentence level checker against supplied
  source context. Same request shape as Latent's public SDK contract
  (prompt, answer, context, back a verdict plus per sentence risk),
  read from the runlatent 0.3.9 wheel on PyPI. It needs no API key,
  no GPU and no network.
* `server.py`: stdlib HTTP sidecar. POST /score, GET /queue,
  POST /review, GET /metrics, GET /export (labeled JSONL for
  fine-tuning), GET /health.
* `queue.py`: review queue. Flagged answers wait for a human decision;
  each decision becomes one labeled example.
* `demo/hold_demo.py`: the hold policy from Latent's own site
  (flagged answer replaced with safe text) on scripted answers.

## Verification (real numbers, 2026-10-25)

* Tests: 9/9 passing, stdlib unittest only.
  Run: `python3 -m unittest discover -s tests -v`
* Benchmark: 24 fixed cases (12 clean, 12 planted hallucinations),
  `benchmark/results.json`. Precision 1.0, recall 0.75.
  Per kind escalated: unsupported number 3/3, wrong date 3/3,
  invented name 3/3, unsupported claim 0/3 escalated but 3/3 surfaced
  at sentence level (risk 0.6, below the 0.7 escalation threshold).
  Mean local scorer latency 0.136 ms. That latency is for this local
  checker only and is not comparable to Latent's reader.
* Live check of Latent's real SDK: runlatent 0.3.9 downloaded from
  PyPI and its /score contract inspected. Their cloud API was NOT
  called: no API key exists here, and none was invented.

## Honest gaps

* This checker only catches hallucinations that reuse a number, date
  or name absent from the context, or a claim with no shared content
  words. A wrong answer built only from in-context facts will pass.
  That is the same class of limit Latent's own research notes flag
  for probe based detection.
* The benchmark is a 24 case fixture set written for this build, not
  a general accuracy claim. The recall gap (unsupported claims) is
  real and is stated in the outreach, not hidden.
* No Latent API key, no GPU, no live model calls: answers in the demo
  and benchmark are fixed strings. The next step that needs her is
  a Latent dashboard key if she wants the real reader in the loop.
* Team size 1, San Francisco, no announced seed beyond YC's standard
  investment, no remote or India statement, pay undisclosed. The ask
  in the outreach is a 20 minute call or a 2 week trial, not a job
  application to a posted role.

## Status

Nothing sent, posted or pushed. Her to dos: create the GitHub repo herself
and share the URL, send the email to founders@runlatent.ai. No video: she
decided 2026-10-27 to record no more videos; offer a live walkthrough instead.
LOOM_SCRIPT.md is superseded.
