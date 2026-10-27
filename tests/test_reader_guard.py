"""Tests for the real-reader wrapper, guardrail, and hash-only queue.

The runlatent SDK is stubbed here: no key exists in this environment,
and these tests must never call the network. The stub uses the real
CheckResult shape from the runlatent 0.3.9 wheel: scored, verdict,
risk, sentences with start/end/text/risk, threshold, suite_fired.
"""

import json
import os
import sys
import tempfile
import types
import unittest
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latent_guard.guard import guard_answer, guard_stream  # noqa: E402
from latent_guard.queue import ReviewQueue  # noqa: E402
from latent_guard.reader import LatentReader  # noqa: E402
from latent_guard.scorer import score_answer  # noqa: E402

CTX = "Revenue was $4.2 million. Priya Sharma is the fund manager."


class _Sentence:
    def __init__(self, start, end, text, risk):
        self.start, self.end, self.text, self.risk = start, end, text, risk


class _Result:
    def __init__(self, scored=True, verdict="escalate", risk=0.9, sentences=None,
                 threshold=0.7, status=200, detail=None):
        self.scored, self.verdict, self.risk = scored, verdict, risk
        self.sentences = sentences
        self.threshold = threshold
        self.suite_fired = ("grounding",) if scored else ()
        self.detail, self.status = detail, status
        self.request_id = "req-1"
        self.resume = None
        self.retry_after_s = None


def _install_stub(verdict="escalate", risk=0.9, sentences=None, fail=None):
    module = types.ModuleType("runlatent")
    calls = []

    def check(prompt, answer, context=None, timeout=2.0, detail=True):
        if fail is not None:
            raise fail
        return _Result(verdict=verdict, risk=risk, sentences=sentences)

    def configure(api_url=None, api_key=None, **kwargs):
        calls.append((api_url, api_key))

    module.check = check
    module.configure = configure
    module.configure_calls = calls
    sys.modules["runlatent"] = module
    os.environ["LATENT_API_URL"] = "https://example.invalid"
    os.environ["LATENT_API_KEY"] = "test-key"
    return module


class TestReader(unittest.TestCase):
    def setUp(self):
        self._saved = dict(os.environ)
        self._saved_module = sys.modules.pop("runlatent", None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved)
        if self._saved_module is not None:
            sys.modules["runlatent"] = self._saved_module
        else:
            sys.modules.pop("runlatent", None)

    def test_reader_keeps_offsets_and_verdict(self):
        sentences = [_Sentence(0, 24, "Revenue was $9.9 million.", 0.9)]
        _install_stub(sentences=sentences)
        reader = LatentReader()
        self.assertTrue(reader.configured)
        read = reader.read("Summarize.", "Revenue was $9.9 million.", CTX)
        self.assertTrue(read["scored"])
        self.assertEqual(read["verdict"], "escalate")
        self.assertEqual(read["sentences"][0]["start"], 0)
        self.assertEqual(read["sentences"][0]["risk"], 0.9)

    def test_reader_refusal_is_unknown_not_a_pass(self):
        _install_stub(fail=ConnectionError("no route"))
        reader = LatentReader()
        read = reader.read("Summarize.", "Revenue was $9.9 million.", CTX)
        self.assertFalse(read["scored"])
        self.assertEqual(read["verdict"], "unknown")
        self.assertEqual(read["unscored_reason"], "read_failed")

    def test_reader_not_configured_without_key(self):
        os.environ.pop("LATENT_API_URL", None)
        os.environ.pop("LATENT_API_KEY", None)
        reader = LatentReader()
        self.assertFalse(reader.configured)
        read = reader.read("Summarize.", "Revenue was $9.9 million.", CTX)
        self.assertEqual(read["unscored_reason"], "not_configured")

    def test_explicit_args_reach_sdk_configure(self):
        stub = _install_stub()
        reader = LatentReader(api_url="https://explicit.example",
                              api_key="explicit-key")
        self.assertIn(("https://explicit.example", "explicit-key"),
                      stub.configure_calls)
        self.assertEqual(reader.api_url, "https://explicit.example")


class TestGuard(unittest.TestCase):
    def setUp(self):
        self._saved = dict(os.environ)
        self._saved_module = sys.modules.pop("runlatent", None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved)
        if self._saved_module is not None:
            sys.modules["runlatent"] = self._saved_module
        else:
            sys.modules.pop("runlatent", None)

    def test_hold_replaces_flagged_answer(self):
        _install_stub()
        guard = guard_answer("Summarize.", "Revenue was $9.9 million.", CTX,
                             reader=LatentReader(), policy="hold")
        self.assertEqual(guard["action"], "held")
        self.assertIn("holding it", guard["delivered_answer"])
        self.assertEqual(guard["source"], "reader")

    def test_reroute_uses_stronger_model(self):
        _install_stub()
        guard = guard_answer("Summarize.", "Revenue was $9.9 million.", CTX,
                             reader=LatentReader(), policy="reroute",
                             stronger_model=lambda p, a: "Revenue was $4.2 million.")
        self.assertEqual(guard["action"], "rerouted")
        self.assertEqual(guard["delivered_answer"], "Revenue was $4.2 million.")

    def test_unscored_fail_open_and_closed(self):
        _install_stub(fail=ConnectionError("no route"))
        reader = LatentReader()
        open_guard = guard_answer("Summarize.", "Revenue was $9.9 million.", CTX,
                                  reader=reader, fail_closed=False)
        self.assertEqual(open_guard["action"], "unscored_delivered")
        closed_guard = guard_answer("Summarize.", "Revenue was $9.9 million.", CTX,
                                    reader=reader, fail_closed=True)
        self.assertEqual(closed_guard["action"], "unscored_held")

    def test_local_prefilter_is_labeled_local(self):
        guard = guard_answer("Summarize.", "Revenue was $9.9 million.", CTX,
                             local_scorer=score_answer, policy="hold")
        self.assertEqual(guard["source"], "local")
        self.assertEqual(guard["action"], "held")

    def test_stream_cuts_at_first_flagged_sentence(self):
        sentences = [
            _Sentence(0, 24, "Revenue was $4.2 million.", 0.1),
            _Sentence(25, 49, "Revenue was $9.9 million.", 0.9),
        ]
        _install_stub(sentences=sentences)
        result = guard_stream("Summarize.",
                              ["Revenue was $4.2 million. Revenue was $9.9 million."],
                              CTX, reader=LatentReader())
        self.assertEqual(result["action"], "cut")
        self.assertEqual(result["cut_at"], 1)
        self.assertEqual(result["delivered_answer"], "Revenue was $4.2 million.")

    def test_stream_cuts_flagged_answer_with_no_sentence_rows(self):
        _install_stub(sentences=None)
        result = guard_stream("Summarize.", ["Revenue was $9.9 million."],
                              CTX, reader=LatentReader())
        self.assertEqual(result["action"], "cut")
        self.assertEqual(result["cut_at"], 0)
        self.assertEqual(result["delivered_answer"], "")

    def test_stream_fail_closed_with_no_reader_holds_everything(self):
        # No reader and fail-closed: the action says held, so nothing
        # may be delivered. (Fail-open delivers the full answer.)
        held = guard_stream("Summarize.", ["Revenue was $9.9 million."],
                            CTX, reader=None, fail_closed=True)
        self.assertEqual(held["action"], "unscored_held")
        self.assertEqual(held["delivered_answer"], "")
        open_result = guard_stream("Summarize.", ["Revenue was $9.9 million."],
                                   CTX, reader=None, fail_closed=False)
        self.assertEqual(open_result["action"], "unscored_delivered")
        self.assertEqual(open_result["delivered_answer"], "Revenue was $9.9 million.")


class TestHashQueue(unittest.TestCase):
    def test_hash_mode_stores_no_text(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "q.json")
            queue = ReviewQueue(path, store_text=False)
            result = score_answer("Summarize.", "Revenue was $9.9 million.", CTX)
            entry = queue.add(result, "Summarize the secret report.", "Revenue was $9.9 million.")
            self.assertNotIn("9.9", entry["answer"])
            self.assertEqual(len(entry["answer_sha256"]), 64)
            blob = open(path, encoding="utf-8").read()
            self.assertNotIn("9.9", blob)
            self.assertNotIn("secret report", blob)
            self.assertNotIn("Revenue was", blob)
            queue.review(result["request_id"], "confirmed")
            self.assertEqual(queue.labeled_examples()[0]["label"], "hallucinated")


class TestServerAlias(unittest.TestCase):
    def test_output_alias_and_offsets(self):
        import threading
        from http.server import ThreadingHTTPServer

        import latent_guard.server as server_mod
        from latent_guard.server import Handler

        with tempfile.TemporaryDirectory() as d:
            server_mod.QUEUE = ReviewQueue(os.path.join(d, "q.json"))
            srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            req = urllib.request.Request(
                base + "/score", method="POST",
                data=json.dumps({"prompt": "Summarize.", "output": "Revenue was $9.9 million.",
                                 "context": CTX}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as resp:
                scored = json.loads(resp.read().decode())
            self.assertEqual(scored["verdict"], "escalate")
            self.assertEqual(scored["source"], "local")
            self.assertEqual(scored["sentences"][0]["start"], 0)
            srv.shutdown()


if __name__ == "__main__":
    unittest.main()
