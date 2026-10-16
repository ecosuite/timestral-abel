import tempfile
import unittest
from pathlib import Path

from timestral_abel import config


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "config.toml"

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file_gives_defaults(self):
        cfg = config.load(self.path)
        self.assertEqual(cfg.listen_port, 18320)
        self.assertEqual(cfg.model.id, "mistralai/Timestral-209M")
        self.assertEqual(cfg.model.revision, "9874d252ab0aae16f7f7c999f98f98d6ad7d8365")
        self.assertEqual(cfg.batching.max_batch_series, 16)
        self.assertEqual(cfg.limits.max_context, 32768)

    def test_example_file_matches_defaults(self):
        example = Path(__file__).resolve().parent.parent / "config.example.toml"
        self.assertEqual(config.load(example), config.Config())

    def test_sections_and_coercion(self):
        self.path.write_text('listen_port = 9000\nnot_a_key = 1\n[model]\ndevice = "cpu"\n'
                             'local_files_only = true\n[batching]\nmax_batch_series = 4\n'
                             '[limits]\nmax_prediction_length = 96\n[warmup]\nseries = 0\n')
        with self.assertLogs("timestral_abel.config", "WARNING"):
            cfg = config.load(self.path)
        self.assertEqual(cfg.listen_port, 9000)
        self.assertEqual(cfg.model.device, "cpu")
        self.assertTrue(cfg.model.local_files_only)
        self.assertEqual(cfg.batching.max_batch_series, 4)
        self.assertEqual(cfg.limits.max_prediction_length, 96)
        self.assertEqual(cfg.warmup.series, 0)

    def test_rejects_bad_values(self):
        cases = ["[batching]\nmax_batch_series = 0\n", "[limits]\nmax_context = 40000\n",
                 '[model]\nlocal_files_only = "yes"\n', "[batching]\nmax_queue_series = 10\n",
                 "[batching]\nbatch_wait_ms = -1\n", "[batching]\nmax_queue_values = 1000\n",
                 "[limits]\nmax_parallel_parses = 0\n", "[limits]\nsocket_timeout = 0\n"]
        for text in cases:
            with self.subTest(text=text):
                self.path.write_text(text)
                with self.assertRaises(ValueError):
                    config.load(self.path)


if __name__ == "__main__":
    unittest.main()
