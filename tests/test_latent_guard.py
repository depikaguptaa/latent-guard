"""Tests for latent_guard. Stdlib unittest only, so they run anywhere
with no installs. Run from ~/workspace/yc-latent:
    python3 -m unittest discover -s tests -v
"""

import json
import os
import sys
import tempfile
import unittest
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latent_guard.queue import ReviewQueue  # noqa: E402
from latent_guard.scorer import score_answer  # noqa: E402

CTX = ("Aurora Growth Fund Q3 2026 report. Revenue was $4.2 million, up 12% "
       "from Q2 2026. Priya Sharma is the fund manager. The fund's largest "
       "holding is Northwind Energy.")


class TestScorer(unittest.TestCase):
    def test_clean_passes(self):
        r = score_answer("Summarize.", "Revenue was $4.2 million, up 12% from Q2 2026.", CTX)
        self.assertEqual(r["verdict"], "pass")
        self.assertFalse(r["flagged"])

    def test_unsupported_number_escalates(self):
        r = score_answer("Summarize.", "Revenue was $9.9 million.", CTX)
        self.assertEqual(r["verdict"], "escalate")
        self.assertIn("unsupported_number", [x["kind"] for s in r["sentences"] for x in s["reasons"]])

    def test_wrong_date_escalates(self):
        r = score_answer("Summarize.", "The fund launched on April 9, 2024.", CTX)
        self.assertEqual(r["verdict"], "escalate")

    def test_invented_name_escalates(self):
        r = score_answer("Summarize.", "Rohan Mehta is the fund manager.", CTX)
        self.assertEqual(r["verdict"], "escalate")

    def test_unsupported_claim_is_soft_signal_only(self):
        r = score_answer("Summarize.", "The office moved to a larger floor last month.", CTX)
        self.assertEqual(r["verdict"], "pass")
        self.assertTrue(any(s["risk"] > 0 for s in r["sentences"]))

    def test_empty_answer_passes_without_crash(self):
        r = score_answer("Summarize.", "", CTX)
        self.assertEqual(r["verdict"], "pass")

    def test_only_flagged_sentence_carries_risk(self):
        r = score_answer("Summarize.",
                         "Priya Sharma is the fund manager. Revenue was $9.9 million.", CTX)
        risks = [s["risk"] for s in r["sentences"]]
        self.assertEqual(risks[0], 0.0)
        self.assertGreater(risks[1], 0.0)


class TestQueue(unittest.TestCase):
    def test_empty_queue_file_opens_empty(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "q.json")
            open(path, "w").close()
            q = ReviewQueue(path)
            self.assertEqual(len(q.entries), 0)

    def test_labeled_examples_tolerate_reader_sentences(self):
        with tempfile.TemporaryDirectory() as d:
            q = ReviewQueue(os.path.join(d, "q.json"))
            result = {"request_id": "r1", "verdict": "escalate", "risk": 0.9,
                      "flagged": True,
                      "sentences": [{"start": 0, "end": 10, "text": "Bad.", "risk": 0.9}]}
            q.add(result, "Summarize.", "Bad.")
            q.review("r1", "confirmed")
            examples = q.labeled_examples()
            self.assertEqual(examples[0]["label"], "hallucinated")
            self.assertEqual(examples[0]["failure_kinds"], [])

    def test_review_and_export(self):
        with tempfile.TemporaryDirectory() as d:
            q = ReviewQueue(os.path.join(d, "q.json"))
            r = score_answer("Summarize.", "Revenue was $9.9 million.", CTX)
            q.add(r, "Summarize.", "Revenue was $9.9 million.")
            self.assertEqual(len(q.pending()), 1)
            q.review(r["request_id"], "confirmed", "number not in source")
            self.assertEqual(len(q.pending()), 0)
            examples = q.labeled_examples()
            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0]["label"], "hallucinated")
            self.assertEqual(q.metrics()["precision_on_reviewed"], 1.0)


class TestServer(unittest.TestCase):
    def test_score_queue_review_export_roundtrip(self):
        import threading
        from http.server import ThreadingHTTPServer

        import latent_guard.server as server_mod
        from latent_guard.server import Handler

        with tempfile.TemporaryDirectory() as d:
            server_mod.QUEUE = ReviewQueue(os.path.join(d, "q.json"))
            srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{srv.server_address[1]}"

            def call(method, path, payload=None):
                req = urllib.request.Request(base + path, method=method,
                                             data=json.dumps(payload).encode() if payload else None,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req) as resp:
                    return json.loads(resp.read().decode())

            scored = call("POST", "/score", {"prompt": "Summarize.", "answer": "Revenue was $9.9 million.", "context": CTX})
            self.assertEqual(scored["verdict"], "escalate")
            self.assertEqual(len(call("GET", "/queue")), 1)
            call("POST", "/review", {"request_id": scored["request_id"], "decision": "confirmed"})
            exported = call("GET", "/export")
            self.assertEqual(exported["label"], "hallucinated")
            metrics = call("GET", "/metrics")
            self.assertEqual(metrics["total_scored"], 1)
            self.assertEqual(metrics["precision_on_reviewed"], 1.0)
            srv.shutdown()

    def test_non_object_json_returns_400_not_a_dropped_connection(self):
        import threading
        from http.server import ThreadingHTTPServer

        import latent_guard.server as server_mod
        from latent_guard.server import Handler

        with tempfile.TemporaryDirectory() as d:
            server_mod.QUEUE = ReviewQueue(os.path.join(d, "q.json"))
            srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            req = urllib.request.Request(base + "/score", method="POST",
                                         data=b"[]",
                                         headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req)
            self.assertEqual(ctx.exception.code, 400)
            srv.shutdown()

    def test_non_string_score_fields_return_400(self):
        import threading
        from http.server import ThreadingHTTPServer

        import latent_guard.server as server_mod
        from latent_guard.server import Handler

        with tempfile.TemporaryDirectory() as d:
            server_mod.QUEUE = ReviewQueue(os.path.join(d, "q.json"))
            srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            for payload in ({"answer": 123, "context": CTX},
                            {"answer": "Revenue was $9.9 million.", "context": [1, 2]},
                            {"answer": "Revenue was $9.9 million.", "context": CTX, "prompt": 7}):
                req = urllib.request.Request(base + "/score", method="POST",
                                             data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"})
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(req)
                self.assertEqual(ctx.exception.code, 400)
            srv.shutdown()


if __name__ == "__main__":
    unittest.main()
