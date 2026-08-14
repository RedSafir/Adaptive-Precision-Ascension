import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import unittest
import torch
import torch.nn as nn
from apa import APALinear, APAConfig, APAManager
import tempfile
import json

class SmokeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = APALinear(16, 16, config=config)
        self.linear2 = APALinear(16, 2, config=config)
    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))

class TestSmoke(unittest.TestCase):
    def setUp(self):
        self.config = APAConfig(fp8_simulation_mode=True, device="cpu", check_interval=1)
        
    def test_forced_overflow_escalation(self):
        model = SmokeModel(self.config)
        manager = APAManager(model, config=self.config)
        # Force overflow by setting very large weights
        nn.init.constant_(model.linear1.weight_master, 400.0)
        optimizer = torch.optim.SGD(manager.get_trainable_parameters(), lr=0.1)
        
        escalation_happened = False
        for _ in range(5):
            manager.pre_step()
            optimizer.zero_grad(set_to_none=True)
            x = torch.randn(8, 16)
            loss = model(x).sum()
            loss.backward()
            should_step = manager.post_backward_sync_and_eval()
            if should_step:
                optimizer.step()
            else:
                optimizer.zero_grad(set_to_none=True)
            if model.linear1.level > 0 or model.linear2.level > 0:
                escalation_happened = True
                break
                
        self.assertTrue(escalation_happened)

    def test_normal_training_no_escalation(self):
        model = SmokeModel(self.config)
        manager = APAManager(model, config=self.config)
        nn.init.normal_(model.linear1.weight_master, mean=0, std=0.01)
        nn.init.normal_(model.linear2.weight_master, mean=0, std=0.01)
        optimizer = torch.optim.SGD(manager.get_trainable_parameters(), lr=0.001)
        
        for _ in range(10):
            manager.pre_step()
            optimizer.zero_grad(set_to_none=True)
            x = torch.randn(8, 16) * 0.1
            loss = model(x).sum()
            loss.backward()
            should_step = manager.post_backward_sync_and_eval()
            if should_step:
                optimizer.step()
            else:
                optimizer.zero_grad(set_to_none=True)
            
        self.assertEqual(model.linear1.level, 0)
        self.assertEqual(model.linear2.level, 0)
        
    def test_check_interval_window_accumulation(self):
        """With check_interval=4, amax from step 1 must persist until evaluation at step 4."""
        cfg = APAConfig(fp8_simulation_mode=True, device="cpu", check_interval=4)
        model = SmokeModel(cfg)
        manager = APAManager(model, config=cfg)
        
        # Step 1: inject high amax value
        manager.pre_step()
        model.linear1.track_telemetry(torch.tensor([100000.0]))
        manager.post_backward_sync_and_eval()
        
        # Steps 2-3: normal
        for _ in range(2):
            manager.pre_step()
            model.linear1.track_telemetry(torch.tensor([1.0]))
            manager.post_backward_sync_and_eval()
            
        # Not yet evaluated at step 3
        self.assertEqual(model.linear1.level, 0)
        
        # Step 4: evaluation fires at check_interval boundary
        manager.pre_step()
        model.linear1.track_telemetry(torch.tensor([1.0]))
        manager.post_backward_sync_and_eval()
        
        # Should have escalated due to step 1's high amax surviving
        self.assertGreater(model.linear1.level, 0)
        
    def test_skip_batch_zeros_grad(self):
        model = SmokeModel(self.config)
        manager = APAManager(model, config=self.config)
        optimizer = torch.optim.SGD(manager.get_trainable_parameters(), lr=0.1)
        
        manager.pre_step()
        optimizer.zero_grad(set_to_none=True)
        x = torch.randn(8, 16)
        loss = model(x).sum()
        loss.backward()
        
        # Manually force NaN flag to trigger skip
        model.linear1.gpu_has_nonfinite.fill_(1)
        
        should_step = manager.post_backward_sync_and_eval()
        self.assertFalse(should_step)
            
    def test_escalation_log_file(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            log_file = tmp.name
            
        cfg = APAConfig(fp8_simulation_mode=True, device="cpu", check_interval=1, log_file=log_file)
        model = SmokeModel(cfg)
        manager = APAManager(model, config=cfg)
        
        manager.pre_step()
        # Force nonfinite + high amax to trigger escalation
        model.linear1.gpu_has_nonfinite.fill_(1)
        model.linear1.gpu_amax.fill_(float('inf'))
        manager.post_backward_sync_and_eval()
        
        self.assertTrue(os.path.exists(log_file))
        with open(log_file, "r") as f:
            lines = f.readlines()
            
        self.assertTrue(len(lines) > 0)
        data = json.loads(lines[0])
        self.assertIn("module", data)
        self.assertIn("old_level", data)
        self.assertIn("new_level", data)
        self.assertIn("reason", data)
        
        os.remove(log_file)

if __name__ == '__main__':
    unittest.main()
