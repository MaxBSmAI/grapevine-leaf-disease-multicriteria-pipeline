from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from grape_disease.utils.test_lock import LockedTestAccessError, require_test_access


class TestLockTests(unittest.TestCase):
    def test_locked_record_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "TEST_LOCK.json"
            path.write_text(
                json.dumps({"test_access": False, "reason": "development"}),
                encoding="utf-8",
            )
            with self.assertRaises(LockedTestAccessError):
                require_test_access(path)

    def test_incomplete_unlock_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "TEST_LOCK.json"
            path.write_text(json.dumps({"test_access": True}), encoding="utf-8")
            with self.assertRaises(LockedTestAccessError):
                require_test_access(path)


if __name__ == "__main__":
    unittest.main()
