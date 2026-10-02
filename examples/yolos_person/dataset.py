import os
import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple

class YoloPersonBallDataset(Dataset):
    """Fast PyTorch Dataset for YOLO-formatted object detection datasets

    Loads images and annotations for YOLOS training and evaluation.
    Converts YOLO format (cls, cx, cy, w, h) to Hugging Face YOLOS target dictionaries:
    {'class_labels': LongTensor[N], 'boxes': FloatTensor[N, 4]}
    """
    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        img_size: int = 512,
        class_filter: Optional[List[int]] = None,
        max_samples: Optional[int] = None
    ):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.class_filter = set(class_filter) if class_filter is not None else None

        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        all_files = sorted(os.listdir(img_dir))
        self.img_paths = [
            os.path.join(img_dir, f)
            for f in all_files
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]

        if max_samples is not None and max_samples > 0:
            self.img_paths = self.img_paths[:max_samples]

        # Normalization constants (ImageNet standard used by ViT/YOLOS)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path = self.img_paths[idx]
        bgr = cv2.imread(img_path)
        if bgr is None:
            # Fallback to blank image if corrupted
            rgb = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
            target = {
                'class_labels': torch.zeros((0,), dtype=torch.long),
                'boxes': torch.zeros((0, 4), dtype=torch.float32)
            }
            return rgb, target

        # Resize image to square target size
        if bgr.shape[0] != self.img_size or bgr.shape[1] != self.img_size:
            bgr = cv2.resize(bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        # BGR -> RGB and to float tensor [3, H, W] in [0, 1]
        rgb_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).to(torch.float32) / 255.0
        tensor = (tensor - self.mean) / self.std

        # Load corresponding label file
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(self.label_dir, f"{base_name}.txt")

        classes = []
        boxes = []

        if os.path.isfile(lbl_path):
            with open(lbl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        if self.class_filter is not None and cls_id not in self.class_filter:
                            continue
                        cx, cy, w, h = map(float, parts[1:5])
                        # Clamp normalized box coordinates to [0, 1]
                        cx = min(max(cx, 0.0), 1.0)
                        cy = min(max(cy, 0.0), 1.0)
                        w = min(max(w, 1e-4), 1.0)
                        h = min(max(h, 1e-4), 1.0)
                        classes.append(cls_id)
                        boxes.append([cx, cy, w, h])

        if len(classes) > 0:
            target = {
                'class_labels': torch.tensor(classes, dtype=torch.long),
                'boxes': torch.tensor(boxes, dtype=torch.float32)
            }
        else:
            target = {
                'class_labels': torch.zeros((0,), dtype=torch.long),
                'boxes': torch.zeros((0, 4), dtype=torch.float32)
            }

        return tensor, target

def yolo_collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    """Custom collate function for YOLOS training

    Stacks image tensors into [B, 3, H, W] and formats targets as List[Dict].
    """
    pixel_values = torch.stack([item[0] for item in batch], dim=0)
    labels = [item[1] for item in batch]
    return pixel_values, labels

def build_dataloader(
    img_dir: str,
    label_dir: str,
    batch_size: int = 16,
    img_size: int = 512,
    shuffle: bool = True,
    num_workers: int = 4,
    class_filter: Optional[List[int]] = None,
    max_samples: Optional[int] = None
) -> DataLoader:
    dataset = YoloPersonBallDataset(
        img_dir=img_dir,
        label_dir=label_dir,
        img_size=img_size,
        class_filter=class_filter,
        max_samples=max_samples
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=yolo_collate_fn,
        pin_memory=True,
        drop_last=True
    )
