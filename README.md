# latent_guard: an offline checker on Latent's /score contract

Latent flags the failures that sink AI answers: made up numbers, wrong dates, invented names. This is a small version of that layer, built to Latent's own public contract, that runs anywhere a hosted reader cannot.

It takes the same shape Latent's SDK posts to `/score`: prompt, answer, and source context in, a verdict plus one risk score per sentence out. It needs no API key, no GPU, and no network, so it works air-gapped, or as a cheap first pass that only sends flagged answers on to a stronger judge.

## What it checks

Sentence by sentence, against the supplied context only:

- Unsupported number: a number in the answer that appears nowhere in the context (amounts, percents, counts). Risk 0.90.
- Wrong date: a date in the answer that appears nowhere in the context. Risk 0.85.
- Invented name: a capitalized name in the answer that appears nowhere in the context. Risk 0.85.
- Unsupported claim: a factual sentence that shares no content words, number, date, or name with any context sentence. Risk 0.60, a soft signal, below the escalation threshold on purpose.

Risk 0.70 or above escalates. Below that, the answer passes, with per-sentence risks still reported so nothing is hidden.

## Measured results

24 fixed cases in `benchmark/cases.json` (12 clean, 12 planted hallucinations), run with `python benchmark/run.py`. Fully reproducible, no model, no network, fixed answer strings.

| | |
|---|---|
| Precision | 1.0 (9 escalated, 0 false flags) |
| Recall | 0.75 (9 of 12 hallucinations escalated) |
| Mean scorer latency | 0.14 ms (local checker only, not comparable to Latent's reader) |

Per kind, escalated: unsupported number 3/3, wrong date 3/3, invented name 3/3, unsupported claim 0/3. The unsupported claims were all surfaced at sentence level (risk 0.60), just below the threshold. That gap is real and is stated here, not hidden.

Tests: 9/9 passing, stdlib unittest only. Full rows are in `benchmark/results.json`.

## Try it

```bash
python -m unittest discover -s tests -v
python benchmark/run.py
python demo/hold_demo.py
```

The demo shows Latent's own hold policy from its site: a flagged answer is held and replaced with safe text, a clean answer is delivered.

## Use it as a sidecar

```bash
python -m latent_guard.server   # listens on 127.0.0.1:8765, data in ./data
```

```bash
curl -X POST http://127.0.0.1:8765/score \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Summarize the report.", "answer": "Revenue was $9.9 million.", "context": "Revenue was $4.2 million."}'
```

Response: verdict, risk, per-sentence risks and reasons, plus `queue_status`. Flagged answers wait in a review queue:

- `GET /queue`: flagged answers awaiting a human decision.
- `POST /review`: `{"request_id": "...", "decision": "confirmed" | "released", "note": "..."}`. Each decision becomes one labeled example.
- `GET /metrics`: counts, flag rate, precision on reviewed items.
- `GET /export`: labeled examples as JSON lines, for fine-tuning.
- `GET /health`: uptime check.

Or use it as a library, one pure function, no I/O:

```python
from latent_guard import score_answer

result = score_answer(prompt, answer, context)
# result["verdict"] is "pass" or "escalate"
# result["sentences"] carries risk and reasons per sentence
```

Set `PORT` and `LATENT_GUARD_DATA` to change the port and data directory. No dependencies beyond the standard library.

## Honest limits

1. This checker only catches hallucinations that reuse a number, date, or name absent from the context, or a claim with no shared content words. A wrong answer built only from in-context facts, combined wrongly, will pass. That is the same class of limit Latent's own research notes flag for probe based detection.
2. The benchmark is a 24-case fixture set written for this build, not a general accuracy claim. The recall gap on freeform unsupported claims is real.
3. No Latent API key, no GPU, no live model calls. Answers in the demo and benchmark are fixed strings. Latent's cloud API was not called. The contract was read from the public `runlatent` 0.3.9 wheel on PyPI.
4. No job posting was found for this Latent. This is a founder play, not an application to a posted role. The ask in the outreach is a 20-minute call or a 2-week trial, not a claim about a role.

## What is next

The useful next step is to run this next to the real reader: same prompt, answer, and context through both, compare verdicts on real traffic, and tune the threshold and the unsupported-claim signal on labeled review decisions from the queue. That needs a Latent dashboard key if you want the real reader in the loop. I am happy to run it as a 2-week trial against your reader, or walk through the code live.

Built by Depika Gupta, AI and backend engineer. I build LangChain multi-agent systems, RAG, and LLM observability at a remote US startup. github.com/depikaguptaa
