import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import unittest
import torch
import torch.nn as nn
from apa import APALinear, APAConfig, APAManager
from apa.cuda_graph import APACUDAGraphRunner


class SmallCUDAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = APALinear(32, 32, config=config)
        self.linear2 = APALinear(32, 4, config=config)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


class TestCUDAGraph(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for CUDA Graph tests")
    def test_cuda_graph_capture_and_recapture_on_escalation(self):
        device = torch.device('cuda')
        config = APAConfig(device='cuda', check_interval=1, fp8_simulation_mode=True)
        model = SmallCUDAModel(config).to(device)
        manager = APAManager(model, config=config)
        optimizer = torch.optim.AdamW(manager.get_trainable_parameters(), lr=1e-3, capturable=True)

        sample_x = torch.randn(8, 32, device=device)
        sample_y = torch.randint(0, 4, (8,), device=device)

        runner = APACUDAGraphRunner(
            model=model,
            optimizer=optimizer,
            sample_x=sample_x,
            sample_y=sample_y,
            apa_manager=manager,
            loss_fn=nn.functional.cross_entropy,
            warmup_steps=2
        )

        # Run 3 steps normally
        for _ in range(3):
            loss = runner.step(sample_x, sample_y)
            self.assertIsInstance(loss, float)

        # Force an overflow to trigger dynamic escalation & re-capture
        model.linear1.track_telemetry(torch.tensor([10000.0], device=device))

        # This step should trigger manager._do_full_evaluation(), detect overflow,
        # escalate linear1, and seamlessly re-capture the CUDA graph without CUDACachingAllocator crash
        loss_escalated = runner.step(sample_x, sample_y)
        self.assertIsInstance(loss_escalated, float)
        self.assertGreater(model.linear1.level, 0)

        # Run another step with the re-captured graph to verify it replays properly
        loss_post = runner.step(sample_x, sample_y)
        self.assertIsInstance(loss_post, float)


if __name__ == '__main__':
    unittest.main()
