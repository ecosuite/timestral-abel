import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path

from timestral_abel import auth
from helpers import KEY_A, KEY_B, UPSTREAM, USERS_TOML


class AuthTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "users.toml"

    def tearDown(self):
        self.tmp.cleanup()

    def test_parse_rejects_bad_files(self):
        good = "k" * 20
        cases = [
            {"users": {"bad id!": {"key": good}}},
            {"users": {"a": {"key": "short"}}},
            {"users": {"a": {"key": good}, "b": {"key": good}}},
            {"users": {"a": {"key": good, "name": "  "}}},
            {"users": "nope"},
        ]
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                auth.parse(data)

    def test_keys_and_upstream_key(self):
        self.path.write_text(USERS_TOML)
        d = auth.Directory(self.path)
        self.assertEqual(d.authenticate(KEY_A).id, "alice")
        self.assertEqual(d.authenticate(KEY_A).name, "Alice")
        self.assertEqual(d.authenticate(KEY_B).name, "bob")
        self.assertIsNone(d.authenticate(UPSTREAM))  # the upstream key is never a forecasting key
        self.assertIsNone(d.authenticate("x" * 32))
        self.assertIsNone(d.authenticate(None))
        self.assertIsNone(d.authenticate(""))

    def test_user_with_upstream_key_is_left_out(self):
        self.path.write_text(f'upstream_key = "{UPSTREAM}"\n[users.sneaky]\nkey = "{UPSTREAM}"\n'
                             f'[users.alice]\nkey = "{KEY_A}"\n')
        with self.assertLogs("timestral_abel.auth", "WARNING"):
            d = auth.Directory(self.path)
            self.assertIsNone(d.authenticate(UPSTREAM))
        self.assertEqual(d.authenticate(KEY_A).id, "alice")

    def test_reloads_and_keeps_last_good(self):
        self.path.write_text(USERS_TOML)
        d = auth.Directory(self.path)
        self.assertEqual(d.authenticate(KEY_A).id, "alice")

        time.sleep(0.01)
        self.path.write_text(f'[users.carol]\nname = "C"\nkey = "{"c" * 20}"\n')
        self.assertEqual(d.authenticate("c" * 20).name, "C")
        self.assertIsNone(d.authenticate(KEY_A))

        time.sleep(0.01)
        self.path.write_text("this is [not toml")  # a typo mid-edit
        with self.assertLogs("timestral_abel.auth", "ERROR"):
            self.assertEqual(d.authenticate("c" * 20).id, "carol")

    def test_missing_file_refuses(self):
        with self.assertLogs("timestral_abel.auth", "ERROR"):
            d = auth.Directory(self.path)
            self.assertIsNone(d.authenticate("anything-at-all-here"))

    def test_removed_or_unreadable_file_refuses_old_keys(self):
        self.path.write_text(USERS_TOML)
        d = auth.Directory(self.path)
        self.assertEqual(d.authenticate(KEY_A).id, "alice")
        self.path.unlink()  # an operator cutting off access
        with self.assertLogs("timestral_abel.auth", "ERROR"):
            self.assertIsNone(d.authenticate(KEY_A))
        self.assertIsNone(d.authenticate(KEY_B))
        self.path.write_text(USERS_TOML)  # and back
        self.assertEqual(d.authenticate(KEY_A).id, "alice")
        # Unreadable between stat and open (a permission change, or a race).
        time.sleep(0.01)
        self.path.write_text(USERS_TOML + "\n")
        with mock.patch.object(auth, "load", side_effect=PermissionError("denied")), \
                self.assertLogs("timestral_abel.auth", "ERROR"):
            self.assertIsNone(d.authenticate(KEY_A))

    def test_request_key(self):
        cases = [
            ({"Authorization": "Bearer abc"}, "abc"),
            ({"Authorization": "bearer  abc "}, "abc"),
            ({"x-api-key": "xyz"}, "xyz"),
            ({"Authorization": "Basic abc", "x-api-key": "xyz"}, "xyz"),
            ({"Authorization": "Bearer "}, None),
            ({}, None),
        ]
        for headers, expected in cases:
            with self.subTest(headers=headers):
                self.assertEqual(auth.request_key(headers), expected)


if __name__ == "__main__":
    unittest.main()
