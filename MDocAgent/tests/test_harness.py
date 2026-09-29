import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace


MDOC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if MDOC_ROOT not in sys.path:
    sys.path.insert(0, MDOC_ROOT)

sys.modules.setdefault("torch", SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)))
sys.modules.setdefault("PIL", SimpleNamespace(Image=SimpleNamespace()))
sys.modules.setdefault("PIL.Image", SimpleNamespace())
sys.modules.setdefault("pymupdf", SimpleNamespace())


class FakeSeries(list):
    def mean(self):
        return sum(self) / len(self)


class FakeDataFrame:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        return FakeSeries([row[key] for row in self.rows])


sys.modules.setdefault("pandas", SimpleNamespace(DataFrame=FakeDataFrame))

from agents.base_agent import Agent
from scripts import harness


class HarnessTests(unittest.TestCase):
    def test_has_ground_truth_accepts_custom_key(self):
        samples = [
            {"question": "q1", "gold": "a1"},
            {"question": "q2", "gold": "a2"},
        ]

        self.assertTrue(harness.has_ground_truth(samples, "gold"))
        self.assertFalse(harness.has_ground_truth(samples, "answer"))

    def test_write_summary_records_prediction_and_eval_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = os.path.join(tmpdir, "harness_summary.json")

            written = harness.write_summary(
                summary_path,
                run_name="exp-a",
                result_dir=os.path.join(tmpdir, "results"),
                prediction_path=os.path.join(tmpdir, "pred.json"),
                eval_path=os.path.join(tmpdir, "eval.json"),
                sample_count=2,
                eval_sample_count=2,
                average_binary_correctness=0.5,
                eval_skipped=False,
            )

            self.assertEqual(written, summary_path)
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        self.assertEqual(data["run_name"], "exp-a")
        self.assertEqual(data["sample_count"], 2)
        self.assertEqual(data["eval_sample_count"], 2)
        self.assertEqual(data["average_binary_correctness"], 0.5)
        self.assertFalse(data["eval_skipped"])


class FakeDataset:
    def __init__(self, samples, tmpdir, gt_key="answer"):
        self.samples = samples
        self.config = SimpleNamespace(
            question_key="question",
            gt_key=gt_key,
            result_dir=tmpdir,
        )

    def load_latest_results(self):
        return self.samples, os.path.join(self.config.result_dir, "latest.json")


class AgentEvalDatasetTests(unittest.TestCase):
    def test_eval_dataset_raises_when_ground_truth_key_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = FakeDataset([{"question": "q", "ans_exp": "a"}], tmpdir)
            agent = object.__new__(Agent)
            agent.config = SimpleNamespace(truncate_len=None, ans_key="ans_exp")
            agent.eval = lambda question, answer, gt: {"binary_correctness": 1}

            with self.assertRaisesRegex(ValueError, "missing ground truth key 'answer'"):
                agent.eval_dataset(dataset)

    def test_eval_dataset_returns_summary_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = FakeDataset(
                [{"question": "q", "answer": "gold", "ans_exp": "gold"}],
                tmpdir,
            )
            agent = object.__new__(Agent)
            agent.config = SimpleNamespace(truncate_len=None, ans_key="ans_exp")
            agent.eval = lambda question, answer, gt: {"binary_correctness": 1}

            summary = agent.eval_dataset(dataset)

        self.assertEqual(summary["sample_count"], 1)
        self.assertEqual(summary["average_binary_correctness"], 1.0)
        self.assertTrue(summary["results_path"].endswith("_results.json"))


if __name__ == "__main__":
    unittest.main()
