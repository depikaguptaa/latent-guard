# latent_guard: an offline checker shaped like Latent's check() call

Latent flags the failures that sink AI answers: made up numbers, wrong dates, invented names. This is a small local checker shaped like the Python call `runlatent.check(prompt, answer, context=...)`, that runs anywhere a hosted reader cannot. It is not the reader, and it is not drop-in compatible with Latent's raw `/score` body.

The distinction matters. The SDK's `check()` takes prompt, answer, and context, renders the context as a `Knowledge:` line in the messages, and posts messages, output, and context to `/score`. The sentence read in the reply's detail block carries start and end offsets, text, and risk, with no failure kinds and no reasons. Verdicts are `pass`, `escalate`, or `ood`, plus `unscored` and `unknown` when the read did not happen. This repo's checker invents the failure kinds and reasons, returns only `pass` or `escalate`, and has no unscored path of its own. The real reader is reached through `latent_guard/reader.py`, which wraps `runlatent.check()` and keeps its verdicts, sentence offsets, and unscored handling intact. Use that wrapper for any claim about Latent. Use this checker only as a cheap local pre-filter, and say so.

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
| Mean scorer latency | about 0.14 ms (local checker only, not comparable to Latent's reader) |

Per kind, escalated: unsupported number 3/3, wrong date 3/3, invented name 3/3, unsupported claim 0/3. The unsupported claims were all surfaced at sentence level (risk 0.60), just below the threshold. That gap is real and is stated here, not hidden.

Tests: 24/24 passing, stdlib unittest only, including the reader wrapper, guardrail, hash-only queue, and server alias tests. Full rows are in `benchmark/results.json`.

On the harder 30-case adversarial set in `eval/adversarial_cases.json` (unit conversions, date formats, same-token contradictions, negation, single names, no-source cases), the local checker scores precision 0.67 and recall 0.44: it false-flags both format variants and misses every contradiction and negation. The reader column in `eval/results.json` is not run, because no key exists here. That gap is the reason the checker is a pre-filter only.

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

Response: verdict, risk, per-sentence risks, reasons, and start/end offsets, plus `queue_status`. The reasons and offsets are this shim's own: the real reader's sentences carry start, end, text, and risk only. The sidecar also accepts `output` as an alias for `answer`, matching the raw `/score` body field name. Flagged answers wait in a review queue:

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

1. This checker only catches hallucinations that reuse a number, date, or name absent from the context, or a claim with no shared content words. A wrong answer built only from in-context facts, combined wrongly, will pass.
2. The benchmark is a 24-case fixture set written for this build, not a general accuracy claim. The recall gap on freeform unsupported claims is real.
3. No Latent API key, no GPU, no live model calls. Answers in the demo and benchmark are fixed strings. Latent's cloud API was not called. The contract was read from the public `runlatent` 0.3.9 wheel on PyPI. The review queue also differs from Latent's console in one important way: by default it stores full prompt and answer text in a JSON file. Latent's site says each scored request writes a hash-only record, never the text. Set `LATENT_GUARD_STORE_TEXT=0` for hash-only queue storage in this repo, and use it for anything beyond a local demo.
4. The first version had no fail-open or fail-closed behavior, no streaming cut, no `ood` or `unscored` verdicts, and no action reporting. The guardrail in `latent_guard/guard.py` adds those, on top of the real reader when it is configured, and labels the local checker as a pre-filter only when it is not.
5. No job posting was found for this Latent. This is a founder play, not an application to a posted role. The ask in the outreach is a 20-minute call or a 2-week trial, not a claim about a role.

## The real loop

`eval/run_eval.py` runs the same cases through the local checker and through the real reader, side by side, and refuses to invent the reader's numbers. Without a key it reports the reader column as not run. With a key it compares verdicts, per-kind recall, unscored counts, and latency, and lists every case where the two disagree.

```bash
pip install "runlatent==0.3.9"
export LATENT_API_URL="https://api.runlatent.ai"
export LATENT_API_KEY="your key here"
python eval/run_eval.py
```

`latent_guard/guard.py` is the guardrail to use in a real app: it calls the reader, holds, replaces, or re-routes a flagged answer, and when the read did not happen it follows the fail-open or fail-closed setting you chose, and says which one it followed. The local checker is never presented as the reader's verdict.

Built by Depika Gupta, AI and backend engineer. I build LangChain multi-agent systems, RAG, and LLM observability at a remote US startup. github.com/depikaguptaa
