import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_record_replay_and_compare(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "episode.json"
            run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "supportbench",
                    "run",
                    "--case",
                    "lamp-tracking",
                    "--policy",
                    "context",
                    "--seed",
                    "13",
                    "--output",
                    str(trace_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("Recorded", run.stdout)
            self.assertEqual(json.loads(trace_path.read_text())["case_id"], "lamp-tracking")
            replay = subprocess.run(
                [sys.executable, "-m", "supportbench", "replay", str(trace_path)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("Replay matched", replay.stdout)
            compare = subprocess.run(
                [sys.executable, "-m", "supportbench", "compare", "--seed", "13"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(len(json.loads(compare.stdout)), 9)


if __name__ == "__main__":
    unittest.main()
