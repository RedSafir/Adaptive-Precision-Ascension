import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import unittest
import torch
import torch.nn as nn
from apa import APALinear, APAConfig, APAManager

class TestModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(100, 64)
        self.linear1 = APALinear(64, 128, config=config)
        self.norm = nn.LayerNorm(128)
        self.linear2 = APALinear(128, 10, config=config)
        
    def forward(self, x):
        x = self.embed(x)
        x = self.linear1(x)
        x = self.norm(x)
        x = self.linear2(x)
        return x

class TestParameterRegistry(unittest.TestCase):
    def setUp(self):
        self.config = APAConfig(fp8_simulation_mode=True, device="cpu")
        self.model = TestModel(self.config)
        self.manager = APAManager(self.model, config=self.config)
        # Ensure working copies exist so the assertion inside get_trainable_parameters
        # can correctly compute working buffer count
        self.manager.pre_step()
        self.trainable_params = self.manager.get_trainable_parameters()

    def test_apa_linear_masters_in_trainable(self):
        param_ids = [id(p) for p in self.trainable_params]
        self.assertIn(id(self.model.linear1.weight_master), param_ids)
        self.assertIn(id(self.model.linear1.bias_master), param_ids)
        self.assertIn(id(self.model.linear2.weight_master), param_ids)
        self.assertIn(id(self.model.linear2.bias_master), param_ids)

    def test_layernorm_params_in_trainable(self):
        param_ids = [id(p) for p in self.trainable_params]
        self.assertIn(id(self.model.norm.weight), param_ids)
        self.assertIn(id(self.model.norm.bias), param_ids)

    def test_embedding_params_in_trainable(self):
        param_ids = [id(p) for p in self.trainable_params]
        self.assertIn(id(self.model.embed.weight), param_ids)

    def test_working_copies_not_in_trainable(self):
        param_ids = [id(p) for p in self.trainable_params]
        # working copies may be None or non-Parameter tensors
        if self.model.linear1.weight_work is not None:
            self.assertNotIn(id(self.model.linear1.weight_work), param_ids)
        if self.model.linear1.bias_work is not None:
            self.assertNotIn(id(self.model.linear1.bias_work), param_ids)

    def test_param_count_matches(self):
        # embed.weight(1) + linear1.weight_master+bias_master(2) + norm.weight+bias(2) + linear2.weight_master+bias_master(2) = 7
        self.assertEqual(len(self.trainable_params), 7)

    def test_all_params_require_grad(self):
        for p in self.trainable_params:
            self.assertTrue(p.requires_grad)

if __name__ == '__main__':
    unittest.main()
