import unittest
import torch
import torch.nn as nn
from apa import APAConfig, APAManager
from apa.config import LEVEL_FP8, LEVEL_FP16, LEVEL_TF32
from examples.yolov8_coco.model import create_yolov8_apa, convert_yolov8_to_apa, create_yolo_hybrid_cuda_graph

try:
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

@unittest.skipUnless(HAS_ULTRALYTICS, "ultralytics is not installed")
class TestYOLOv8APA(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = APAConfig(device=self.device, fp8_simulation_mode=(self.device == 'cpu'))

    def test_model_conversion_and_preservation(self):
        yolo = YOLO('yolov8n.yaml')
        model = yolo.model.to(self.device)
        convert_yolov8_to_apa(model, config=self.config, preserve_critical_layers=True)

        manager = APAManager(model, config=self.config)
        self.assertEqual(len(manager.apa_modules), 64)

        # 45 layers in Backbone & Neck should start at FP8
        fp8_count = sum(1 for m in manager.apa_modules.values() if m.level == LEVEL_FP8)
        # 19 layers in Detect head should be preserved at TF32
        tf32_count = sum(1 for m in manager.apa_modules.values() if m.level == LEVEL_TF32)

        self.assertEqual(fp8_count, 45)
        self.assertEqual(tf32_count, 19)

    def test_forward_pass(self):
        yolo, model = create_yolov8_apa('yolov8n.yaml', config=self.config, device=self.device)
        dummy_x = torch.randn(2, 3, 640, 640, device=self.device)
        model.eval()
        with torch.no_grad():
            preds = model(dummy_x)
        self.assertIsNotNone(preds)

    def test_training_step_and_escalation(self):
        cfg = get_cfg()
        cfg.data = 'coco8.yaml'
        cfg.model = 'yolov8n.yaml'
        cfg.batch = 2
        cfg.imgsz = 640
        cfg.device = self.device

        trainer = DetectionTrainer(overrides=cfg)
        trainer.setup_model()
        model = trainer.model.to(self.device)
        model.args = cfg

        convert_yolov8_to_apa(model, config=self.config, preserve_critical_layers=True)
        manager = APAManager(model, config=self.config)
        optimizer = torch.optim.AdamW(manager.get_trainable_parameters(), lr=1e-3)

        data_dict = check_det_dataset(cfg.data)
        dataset = build_yolo_dataset(cfg, data_dict['train'], batch=2, data=data_dict, mode='train', rect=False, stride=32)
        dataloader = build_dataloader(dataset, batch=2, workers=0, shuffle=False)

        model.train()
        for batch in dataloader:
            manager.pre_step()
            optimizer.zero_grad()
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device, non_blocking=True)
            batch['img'] = batch['img'].float() / 255.0

            loss, loss_items = model(batch)
            total_loss = loss.sum()
            self.assertFalse(torch.isnan(total_loss))
            self.assertFalse(torch.isinf(total_loss))

            total_loss.backward()
            accepted = manager.post_backward_sync_and_eval()
            self.assertTrue(accepted)
            optimizer.step()
            break

        # Test escalation upon overflow
        layer0_conv = manager.apa_modules['model.0.conv']
        self.assertEqual(layer0_conv.level, LEVEL_FP8)

        manager.pre_step()
        layer0_conv.gpu_has_nonfinite.fill_(1)
        accepted = manager.post_backward_sync_and_eval()
        self.assertFalse(accepted)
        self.assertEqual(layer0_conv.level, LEVEL_FP16)

    def test_hybrid_cuda_graph_execution(self):
        if not torch.cuda.is_available() or self.device != 'cuda':
            self.skipTest("CUDA required for Hybrid CUDA Graph test")

        cfg = get_cfg()
        cfg.data = 'coco8.yaml'
        cfg.model = 'yolov8n.yaml'
        cfg.batch = 2
        cfg.imgsz = 640
        cfg.device = self.device

        trainer = DetectionTrainer(overrides=cfg)
        trainer.setup_model()
        model = trainer.model.to(self.device)
        model.args = cfg

        convert_yolov8_to_apa(model, config=self.config, preserve_critical_layers=True)
        manager = APAManager(model, config=self.config)
        optimizer = torch.optim.AdamW(manager.get_trainable_parameters(), lr=1e-3)

        sample_img = torch.zeros((2, 3, 640, 640), device=self.device, dtype=torch.float32)
        graphed_backbone_neck, detect_head = create_yolo_hybrid_cuda_graph(model, sample_img, warmup_iters=3)
        self.assertIsNotNone(graphed_backbone_neck)
        self.assertIsNotNone(detect_head)

        data_dict = check_det_dataset(cfg.data)
        dataset = build_yolo_dataset(cfg, data_dict['train'], batch=cfg.batch, data=data_dict, mode='train', rect=False, stride=32)
        dataloader = build_dataloader(dataset, batch=cfg.batch, workers=0, shuffle=False)

        batch = next(iter(dataloader))
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        batch['img'] = batch['img'].float() / 255.0

        manager.pre_step()
        optimizer.zero_grad()

        f0, f1, f2 = graphed_backbone_neck(batch['img'])
        preds = detect_head([f0, f1, f2])
        loss, loss_items = model.loss(batch, preds)
        total_loss = loss.sum()

        self.assertFalse(torch.isnan(total_loss))
        self.assertFalse(torch.isinf(total_loss))

        total_loss.backward()
        accepted = manager.post_backward_sync_and_eval()
        self.assertIsInstance(accepted, bool)
        if accepted:
            optimizer.step()

if __name__ == '__main__':
    unittest.main()
