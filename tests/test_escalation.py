import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import unittest
import torch
from apa import APALinear, APAConfig, APAManager
from apa.config import LEVEL_FP8, LEVEL_FP16, LEVEL_TF32, DTYPE_MAP

class TestEscalation(unittest.TestCase):
    def setUp(self):
        self.config = APAConfig(fp8_simulation_mode=True, device="cpu")
        self.module = APALinear(16, 16, config=self.config)
        self.model = torch.nn.Sequential(self.module)
        self.manager = APAManager(self.model, config=self.config)
        
    def test_initial_level_is_fp8(self):
        self.assertEqual(self.module.level, LEVEL_FP8)

    def test_escalation_fp8_to_fp16(self):
        self.module.level = LEVEL_FP8
        self.manager._escalate_module("0", self.module, "OVERFLOW", 999.0)
        self.assertEqual(self.module.level, LEVEL_FP16)

    def test_escalation_fp16_to_tf32(self):
        self.module.level = LEVEL_FP16
        self.manager._escalate_module("0", self.module, "OVERFLOW", 999.0)
        self.assertEqual(self.module.level, LEVEL_TF32)

    def test_escalation_ceiling_at_tf32(self):
        self.module.level = LEVEL_TF32
        self.manager._escalate_module("0", self.module, "OVERFLOW", 999.0)
        self.assertEqual(self.module.level, LEVEL_TF32)

    def test_escalation_is_permanent_never_decreases(self):
        self.module.level = LEVEL_FP8
        self.manager._escalate_module("0", self.module, "OVERFLOW", 999.0)
        self.assertEqual(self.module.level, LEVEL_FP16)
        # No mechanism to decrease — verify it stays
        self.assertGreaterEqual(self.module.level, LEVEL_FP16)
        
    def test_working_dtype_changes_with_level(self):
        self.module.level = LEVEL_FP8
        # In simulation mode, FP8 working_dtype falls back to float32
        expected_fp8 = DTYPE_MAP[LEVEL_FP8] if DTYPE_MAP[LEVEL_FP8] is not None else torch.float32
        self.assertEqual(self.module.working_dtype, expected_fp8)
        self.module.level = LEVEL_FP16
        self.assertEqual(self.module.working_dtype, torch.float16)
        self.module.level = LEVEL_TF32
        self.assertEqual(self.module.working_dtype, torch.float32)
        
    def test_refresh_working_copy_dtype(self):
        self.module.level = LEVEL_FP16
        self.module.refresh_working_copy()
        self.assertEqual(self.module.weight_work.dtype, torch.float16)
        
        self.module.level = LEVEL_TF32
        self.module.refresh_working_copy()
        self.assertEqual(self.module.weight_work.dtype, torch.float32)

if __name__ == '__main__':
    unittest.main()
