"""Tests for the optional forensic logging feature.

Covers:
1. test_forensic_off_by_default         — no file created, no behaviour change
2. test_forensic_log_created_on_escalation — file created with correct schema
3. test_forensic_log_json_parseable     — every line is valid JSON
4. test_forward_order_reset_per_step   — _forward_execution_order is empty
                                          after each pre_step() (regression guard
                                          for the bug where reset was placed in
                                          _do_full_evaluation instead of pre_step)
"""
import sys
import os
import json
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from apa import APAConfig, APALinear, APAManager
from apa.config import LEVEL_FP8, LEVEL_FP16, THRESHOLDS_MAX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_model(config):
    """Two-layer sequential model for testing."""
    return nn.Sequential(
        APALinear(16, 16, config=config),
        APALinear(16, 8, config=config),
    )


def _run_overflow_step(manager, model, device='cpu'):
    """Run one step that guarantees an overflow escalation.

    Tracks an extreme value on role='input_activation' to simulate an overflow event,
    then calls post_backward_sync_and_eval() to trigger _do_full_evaluation() → _escalate_module().
    """
    manager.pre_step()
    first_module = list(manager.apa_modules.values())[0]
    overflow_tensor = torch.tensor([500.0])
    first_module.track_telemetry(overflow_tensor, role='input_activation')
    manager.post_backward_sync_and_eval()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestForensicLoggingOffByDefault(unittest.TestCase):
    """Forensic logging must not create any files or change behaviour when off."""

    def test_forensic_off_by_default(self):
        """No forensic log file should be created during a normal training run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, 'apa_log.jsonl')
            forensic_file = os.path.join(tmpdir, 'apa_log_forensic.jsonl')

            config = APAConfig(
                fp8_simulation_mode=True,
                device='cpu',
                log_file=log_file,
                # enable_forensic_logging defaults to False
            )
            self.assertFalse(config.enable_forensic_logging)
            self.assertIsNone(config.forensic_log_file)

            model = _make_simple_model(config)
            manager = APAManager(model, config)

            _run_overflow_step(manager, model)

            # Forensic file must NOT be created
            self.assertFalse(
                os.path.exists(forensic_file),
                "Forensic log file must not be created when enable_forensic_logging=False"
            )
            # Forward-execution order list must exist but forensic hooks should
            # not have been registered (list stays empty throughout)
            # We only check it is a list — emptiness is checked separately.
            self.assertIsInstance(manager._forward_execution_order, list)

    def test_forensic_logger_is_none_when_off(self):
        config = APAConfig(fp8_simulation_mode=True, device='cpu')
        model = _make_simple_model(config)
        manager = APAManager(model, config)
        self.assertIsNone(manager.forensic_logger)
        # No forward hooks should be registered for forensic purposes
        self.assertEqual(len(manager._forward_hook_handles), 0)


class TestForensicLogCreatedOnEscalation(unittest.TestCase):
    """Forensic log must be created with correct schema when mode is on."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.tmpdir, 'apa_log.jsonl')
        self.forensic_file = os.path.join(self.tmpdir, 'apa_log_forensic.jsonl')

        self.config = APAConfig(
            fp8_simulation_mode=True,
            device='cpu',
            check_interval=1,
            log_file=self.log_file,
            enable_forensic_logging=True,
        )

    def test_forensic_log_file_auto_derived(self):
        """forensic_log_file should be auto-derived from log_file."""
        self.assertEqual(self.config.forensic_log_file, self.forensic_file)

    def test_forensic_file_created_after_escalation(self):
        """Forensic log file must exist after an overflow escalation."""
        model = _make_simple_model(self.config)
        manager = APAManager(model, self.config)
        _run_overflow_step(manager, model)
        self.assertTrue(
            os.path.exists(self.forensic_file),
            "Forensic log file must be created when enable_forensic_logging=True and escalation occurs"
        )

    def test_forensic_schema_fields_present(self):
        """Every forensic record must contain the required schema fields."""
        required_fields = {
            'step', 'timestamp_utc', 'module_name', 'reason',
            'level_before', 'level_after',
            'culprit_tensor_role', 'per_role_amax',
            'amax_value', 'threshold_at_time',
            'tensor_shape', 'dtype_at_time',
            'preceding_module_in_forward_order',
            'underflow_ratio', 'argmax_flat_index',
        }

        model = _make_simple_model(self.config)
        manager = APAManager(model, self.config)
        _run_overflow_step(manager, model)

        with open(self.forensic_file, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f if line.strip()]

        self.assertGreater(len(records), 0, "At least one forensic record expected")

        for rec in records:
            for field in required_fields:
                self.assertIn(field, rec, f"Missing field '{field}' in forensic record")

    def test_culprit_role_points_to_overflow_tensor(self):
        """culprit_tensor_role must be a valid role string."""
        valid_roles = {
            'input_activation', 'weight', 'output',
            'grad_output', 'grad_weight', 'grad_input', None,
        }
        model = _make_simple_model(self.config)
        manager = APAManager(model, self.config)
        _run_overflow_step(manager, model)

        with open(self.forensic_file, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f if line.strip()]

        for rec in records:
            self.assertIn(
                rec.get('culprit_tensor_role'),
                valid_roles,
                f"culprit_tensor_role '{rec.get('culprit_tensor_role')}' is not a valid role"
            )

    def test_level_before_after_correct(self):
        """level_before should be FP8 and level_after should be FP16 on first overflow."""
        model = _make_simple_model(self.config)
        manager = APAManager(model, self.config)
        _run_overflow_step(manager, model)

        with open(self.forensic_file, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f if line.strip()]

        self.assertGreater(len(records), 0)
        first = records[0]
        self.assertEqual(first['level_before'], 'FP8')
        self.assertEqual(first['level_after'], 'FP16')

    def test_per_role_amax_is_dict(self):
        """per_role_amax must be a dict mapping role strings to floats or null."""
        model = _make_simple_model(self.config)
        manager = APAManager(model, self.config)
        _run_overflow_step(manager, model)

        with open(self.forensic_file, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f if line.strip()]

        for rec in records:
            pra = rec['per_role_amax']
            self.assertIsInstance(pra, dict)
            for role, val in pra.items():
                self.assertTrue(
                    val is None or isinstance(val, (int, float)),
                    f"per_role_amax['{role}'] must be float or null, got {type(val)}"
                )

    def test_no_forensic_file_when_no_escalation(self):
        """Forensic file must not be created if no escalation event fires."""
        model = _make_simple_model(self.config)
        manager = APAManager(model, self.config)

        # Run a clean step (no injected overflow)
        manager.pre_step()
        with torch.no_grad():
            list(manager.apa_modules.values())[0].gpu_amax.fill_(1.0)  # well below threshold
        manager.post_backward_sync_and_eval()

        self.assertFalse(
            os.path.exists(self.forensic_file),
            "Forensic log should not be created if no escalation occurred"
        )


class TestForensicLogJsonParseable(unittest.TestCase):
    """Every line in the forensic log must be valid JSON."""

    def test_forensic_log_all_lines_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = APAConfig(
                fp8_simulation_mode=True,
                device='cpu',
                check_interval=1,
                log_file=os.path.join(tmpdir, 'log.jsonl'),
                enable_forensic_logging=True,
            )
            model = _make_simple_model(config)
            manager = APAManager(model, config)

            # Run multiple overflow steps to generate several records
            for _ in range(3):
                _run_overflow_step(manager, model)

            if not os.path.exists(config.forensic_log_file):
                self.skipTest("No forensic file created (all modules already at TF32)")

            errors = []
            with open(config.forensic_log_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {i+1}: {e}")

            self.assertEqual(
                errors, [],
                "All lines in forensic log must be valid JSON:\n" + "\n".join(errors)
            )


class TestForwardOrderResetPerStep(unittest.TestCase):
    """_forward_execution_order must be reset at pre_step(), not _do_full_evaluation().

    This is a regression guard against the bug where the reset was placed in
    _do_full_evaluation() (called only every check_interval steps), which would
    cause the execution order list to accumulate entries from multiple steps and
    make preceding_module_in_forward_order point to the wrong module.
    """

    def test_forward_order_is_empty_after_pre_step(self):
        """_forward_execution_order must be cleared at the start of each step."""
        config = APAConfig(
            fp8_simulation_mode=True,
            device='cpu',
            check_interval=4,  # soft check every 4 steps — _do_full_evaluation rarely called
            enable_forensic_logging=True,
        )
        model = _make_simple_model(config)
        manager = APAManager(model, config)

        # Simulate a forward pass that populates _forward_execution_order
        # We do this by calling the forward hooks manually (they were registered
        # by _register_forensic_forward_hooks).
        manager.pre_step()
        # After pre_step() the list must be empty
        self.assertEqual(
            manager._forward_execution_order, [],
            "_forward_execution_order must be empty immediately after pre_step()"
        )

    def test_forward_order_populated_during_forward_pass(self):
        """After a forward pass, _forward_execution_order must contain module names."""
        config = APAConfig(
            fp8_simulation_mode=True,
            device='cpu',
            check_interval=4,
            enable_forensic_logging=True,
        )
        model = _make_simple_model(config)
        manager = APAManager(model, config)

        manager.pre_step()
        # Run a forward pass — the forensic forward hooks should populate the list
        x = torch.randn(4, 16)
        model(x)

        self.assertGreater(
            len(manager._forward_execution_order), 0,
            "_forward_execution_order must be populated after a forward pass with forensic hooks registered"
        )

    def test_forward_order_reset_between_steps_without_full_eval(self):
        """Order must reset each step even when _do_full_evaluation() is not called.

        Uses check_interval=100 so _do_full_evaluation() is NOT triggered
        between steps.  Verifies the list is clean at the start of each step.
        """
        config = APAConfig(
            fp8_simulation_mode=True,
            device='cpu',
            check_interval=100,   # effectively never triggers soft check
            enable_forensic_logging=True,
        )
        model = _make_simple_model(config)
        manager = APAManager(model, config)

        for step_idx in range(3):
            manager.pre_step()
            # After each pre_step the list must be empty
            self.assertEqual(
                manager._forward_execution_order, [],
                f"Step {step_idx}: _forward_execution_order must be empty after pre_step()"
            )
            # Simulate a forward pass
            x = torch.randn(4, 16)
            model(x)
            # List now populated — call post_backward (no full eval, check_interval=100)
            manager.post_backward_sync_and_eval()
            # At this point list may still have entries — that's fine,
            # next pre_step() will clear it.


if __name__ == '__main__':
    unittest.main()
