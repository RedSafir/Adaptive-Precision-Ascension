import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import unittest
import torch
from apa import APALinear, APAConfig, APAManager
from apa.telemetry import compute_underflow_ratio
from apa.config import THRESHOLDS_MIN, LEVEL_FP8

class TestUnderflow(unittest.TestCase):
    def setUp(self):
        self.config = APAConfig(fp8_simulation_mode=True, device="cpu")
        self.module = APALinear(16, 16, config=self.config)
        self.model = torch.nn.Sequential(self.module)
        self.manager = APAManager(self.model, config=self.config)
        self.vmin = THRESHOLDS_MIN[LEVEL_FP8]

    def test_underflow_ratio_all_below_vmin(self):
        tensor = torch.full((10,), self.vmin / 2.0)
        ratio = compute_underflow_ratio(tensor, self.vmin)
        self.assertAlmostEqual(ratio.item(), 1.0)
        
    def test_underflow_ratio_none_below_vmin(self):
        tensor = torch.full((10,), self.vmin * 2.0)
        ratio = compute_underflow_ratio(tensor, self.vmin)
        self.assertAlmostEqual(ratio.item(), 0.0)
        
    def test_underflow_ratio_half_below(self):
        tensor_below = torch.full((5,), self.vmin / 2.0)
        tensor_above = torch.full((5,), self.vmin * 2.0)
        tensor = torch.cat([tensor_below, tensor_above])
        ratio = compute_underflow_ratio(tensor, self.vmin)
        self.assertAlmostEqual(ratio.item(), 0.5)
        
    def test_underflow_ratio_ignores_zeros(self):
        tensor_below = torch.full((5,), self.vmin / 2.0)
        tensor_zeros = torch.zeros((5,))
        tensor = torch.cat([tensor_below, tensor_zeros])
        ratio = compute_underflow_ratio(tensor, self.vmin)
        self.assertAlmostEqual(ratio.item(), 1.0)
        
    def test_ema_smoothing_convergence(self):
        # Simulate consistent underflow by calling update_underflow_metric with a tensor
        # where all non-zero elements are below V_min
        underflow_tensor = torch.full((100,), self.vmin / 2.0)
        for _ in range(50):
            self.module.update_underflow_metric(underflow_tensor)
        # After many iterations, the gpu_underflow_ratio should be close to 1.0
        # (running max of ratio where all elements underflow)
        self.assertGreaterEqual(self.module.gpu_underflow_ratio.item(), 0.9)
        
    def test_underflow_escalates_without_skip(self):
        """Underflow triggers escalation but NOT batch skip."""
        self.manager.pre_step()
        # Force high underflow ratio directly by setting EMA
        self.module.ema_underflow_ratio = 0.0
        underflow_tensor = torch.full((100,), self.vmin / 2.0)
        for _ in range(50):
            self.module.update_underflow_metric(underflow_tensor)
        # Set ema high enough to trigger
        self.module.ema_underflow_ratio = self.config.theta_underflow + 0.1
        # Also set the gpu_underflow_ratio high for the full eval
        self.module.gpu_underflow_ratio.fill_(self.config.theta_underflow + 0.1)
        
        should_step = self.manager.post_backward_sync_and_eval()
        # Underflow escalation does NOT skip batch
        self.assertTrue(should_step)

if __name__ == '__main__':
    unittest.main()
