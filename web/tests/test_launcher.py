"""Regression for POSIX venv interpreters that share the system binary."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import _common


class LauncherTests(unittest.TestCase):
    def test_executable_match_does_not_replace_environment_identity(self):
        with patch.object(Path, "is_file", return_value=True), \
             patch.object(sys, "prefix", str(_common.ROOT / "not-the-venv")), \
             patch.object(sys, "executable", str(_common.PYTHON)), \
             patch.object(_common.subprocess, "call", return_value=0) as launch:
            with self.assertRaises(SystemExit):
                _common.in_venv()
            launch.assert_called_once()
            self.assertEqual(launch.call_args.args[0][0], str(_common.PYTHON))

    def test_correct_prefix_needs_no_reexec(self):
        with patch.object(Path, "is_file", return_value=True), \
             patch.object(sys, "prefix", str(_common.ROOT / ".venv")), \
             patch.object(sys, "path", list(sys.path)), \
             patch.object(_common.subprocess, "call") as launch:
            _common.in_venv()
            launch.assert_not_called()
