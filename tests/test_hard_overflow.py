import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import unittest
import torch
from apa import APALinear, APAConfig, APAManager
from apa.config import LEVEL_FP8

class TestHardOverflow(unittest.TestCase):
    def setUp(self):
        self.config = APAConfig(fp8_simulation_mode=True, device="cpu")
        self.module = APALinear(16, 16, config=self.config)
        self.model = torch.nn.Sequential(self.module)
        self.manager = APAManager(self.model, config=self.config)
        
    def test_nan_sets_nonfinite_flag(self):
        tensor = torch.tensor([1.0, float('nan'), 3.0])
        self.module.track_telemetry(tensor)
        self.assertTrue(self.module.gpu_has_nonfinite.item() > 0)
        
    def test_inf_sets_nonfinite_flag(self):
        tensor = torch.tensor([1.0, float('inf'), 3.0])
        self.module.track_telemetry(tensor)
        self.assertTrue(self.module.gpu_has_nonfinite.item() > 0)
        
    def test_neg_inf_sets_nonfinite_flag(self):
        tensor = torch.tensor([1.0, float('-inf'), 3.0])
        self.module.track_telemetry(tensor)
        self.assertTrue(self.module.gpu_has_nonfinite.item() > 0)
        
    def test_normal_values_no_flag(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        self.module.track_telemetry(tensor)
        self.assertEqual(self.module.gpu_has_nonfinite.item(), 0)
        
    def test_flag_accumulates_within_step(self):
        self.module.track_telemetry(torch.tensor([1.0, 2.0]))
        self.module.track_telemetry(torch.tensor([float('nan')]))
        self.module.track_telemetry(torch.tensor([3.0, 4.0]))
        self.assertTrue(self.module.gpu_has_nonfinite.item() > 0)
        
    def test_pre_step_resets_nonfinite(self):
        self.module.track_telemetry(torch.tensor([float('nan')]))
        self.manager.pre_step()
        self.assertEqual(self.module.gpu_has_nonfinite.item(), 0)
        
    def test_pre_step_does_not_reset_amax(self):
        self.module.track_telemetry(torch.tensor([100.0, -200.0]))
        self.assertTrue(self.module.gpu_amax.item() >= 200.0)
        self.manager.pre_step()
        # amax is window-persistent — must survive pre_step
        self.assertTrue(self.module.gpu_amax.item() >= 200.0)
        
    def test_hard_overflow_triggers_skip(self):
        self.manager.pre_step()
        tensor = torch.tensor([1.0, float('nan')])
        self.module.track_telemetry(tensor)
        should_step = self.manager.post_backward_sync_and_eval()
        self.assertFalse(should_step)
        
if __name__ == '__main__':
    unittest.main()
