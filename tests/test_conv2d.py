import unittest
import torch
import torch.nn as nn
from apa import APAConv2d, APALinear, APAManager, APAConfig
from apa.config import LEVEL_FP8, LEVEL_FP16, LEVEL_TF32

class TestAPAConv2d(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = APAConfig(device=self.device, fp8_simulation_mode=(self.device == 'cpu'))

    def test_construction_and_from_conv2d(self):
        conv = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=True)
        apa_conv = APAConv2d.from_conv2d(conv, config=self.config)

        self.assertEqual(apa_conv.in_channels, 16)
        self.assertEqual(apa_conv.out_channels, 32)
        self.assertEqual(apa_conv.kernel_size, (3, 3))
        self.assertEqual(apa_conv.stride, (2, 2))
        self.assertEqual(apa_conv.padding, (1, 1))
        self.assertTrue(torch.allclose(apa_conv.weight_master, conv.weight.data.float()))
        self.assertTrue(torch.allclose(apa_conv.bias_master, conv.bias.data.float()))

    def test_forward_backward_all_levels(self):
        for lvl_name, lvl in [('FP8', LEVEL_FP8), ('FP16', LEVEL_FP16), ('TF32', LEVEL_TF32)]:
            conv = APAConv2d(8, 16, kernel_size=3, padding=1, config=self.config).to(self.device)
            conv.level = lvl
            conv.refresh_working_copy()

            x = torch.randn(2, 8, 16, 16, device=self.device, requires_grad=True)
            out = conv(x)

            self.assertEqual(out.shape, (2, 16, 16, 16))
            loss = out.sum()
            loss.backward()

            self.assertIsNotNone(conv.weight_master.grad)
            self.assertFalse(torch.isnan(conv.weight_master.grad).any())
            self.assertFalse(torch.isinf(conv.weight_master.grad).any())

    def test_1x1_and_grouped_conv(self):
        # 1x1 pointwise conv
        conv1x1 = APAConv2d(16, 32, kernel_size=1, stride=1, padding=0, config=self.config).to(self.device)
        x1 = torch.randn(2, 16, 8, 8, device=self.device)
        out1 = conv1x1(x1)
        self.assertEqual(out1.shape, (2, 32, 8, 8))

        # Grouped conv
        conv_grp = APAConv2d(16, 32, kernel_size=3, padding=1, groups=2, config=self.config).to(self.device)
        x2 = torch.randn(2, 16, 8, 8, device=self.device)
        out2 = conv_grp(x2)
        self.assertEqual(out2.shape, (2, 32, 8, 8))

    def test_manager_integration_and_escalation(self):
        model = nn.Sequential(
            APAConv2d(3, 16, kernel_size=3, padding=1, config=self.config),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            APAConv2d(16, 32, kernel_size=3, padding=1, config=self.config),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            APALinear(32, 10, config=self.config)
        ).to(self.device)

        manager = APAManager(model, self.config)
        self.assertEqual(len(manager.apa_modules), 3)

        params = manager.get_trainable_parameters()
        expected_params = [p for p in model.parameters() if p.requires_grad]
        self.assertEqual(len(params), len(expected_params))

        optimizer = torch.optim.AdamW(params, lr=1e-3)

        # Standard step
        manager.pre_step()
        optimizer.zero_grad()
        x = torch.randn(4, 3, 16, 16, device=self.device)
        y = torch.randint(0, 10, (4,), device=self.device)
        out = model(x)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()
        accepted = manager.post_backward_sync_and_eval()
        self.assertTrue(accepted)
        optimizer.step()

        # Hard overflow escalation test
        c1 = manager.apa_modules['0']
        self.assertEqual(c1.level, LEVEL_FP8)
        manager.pre_step()
        c1.gpu_has_nonfinite.fill_(1)
        accepted = manager.post_backward_sync_and_eval()
        self.assertFalse(accepted)
        self.assertEqual(c1.level, LEVEL_FP16)

        # Second overflow escalation test to TF32
        manager.pre_step()
        c1.gpu_has_nonfinite.fill_(1)
        accepted = manager.post_backward_sync_and_eval()
        self.assertFalse(accepted)
        self.assertEqual(c1.level, LEVEL_TF32)

if __name__ == '__main__':
    unittest.main()
