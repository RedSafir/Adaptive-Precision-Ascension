"""
ImageNet-100 Dataset Loader via Hugging Face Hub (clane9/imagenet-100)

Handles authentication via HF token, downloading, caching, and multi-worker
DataLoader construction with standard ImageNet 224x224 data augmentations.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

class ImageNet100HFDataset(Dataset):
    """PyTorch Dataset wrapper around Hugging Face Dataset split."""
    def __init__(self, hf_split, transform=None):
        self.dataset = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert('RGB')
        label = item['label']
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label

def get_hf_token(explicit_token=None):
    """Resolves Hugging Face API token from CLI argument, env var, or cached token."""
    if explicit_token:
        return explicit_token
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if token:
        return token
    try:
        from huggingface_hub import HfFolder
        cached = HfFolder.get_token()
        if cached:
            return cached
    except Exception:
        pass
    return None

def build_transforms(image_size=224):
    """Builds standard ImageNet training and validation data augmentation transforms."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_resize = int((256 / 224) * image_size)
    val_transform = transforms.Compose([
        transforms.Resize(val_resize, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    
    return train_transform, val_transform

def get_imagenet100_loaders(
    batch_size=128,
    hf_token=None,
    data_dir=None,
    num_workers=4,
    image_size=224,
    pin_memory=True
):
    """
    Downloads and prepares DataLoaders for ImageNet-100 from Hugging Face Hub.
    
    Args:
        batch_size: Batch size for training and validation loaders
        hf_token: Optional explicit Hugging Face API token
        data_dir: Optional custom cache directory for downloaded dataset
        num_workers: DataLoader worker processes
        image_size: Target image resolution (default: 224)
        pin_memory: Enable page-locked CUDA memory transfer
        
    Returns:
        (train_loader, val_loader)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Hugging Face `datasets` library is required. "
            "Please run: pip install datasets huggingface_hub"
        )
        
    resolved_token = get_hf_token(hf_token)
    
    print("Loading 'clane9/imagenet-100' from Hugging Face Hub...")
    if resolved_token:
        print("  -> Authenticated with Hugging Face Token.")
    else:
        print("  -> No Hugging Face Token detected; attempting unauthenticated public fetch.")
        
    load_kwargs = {
        "token": resolved_token,
    }
    if data_dir:
        load_kwargs["cache_dir"] = data_dir
        
    hf_ds = load_dataset("clane9/imagenet-100", **load_kwargs)
    
    # Handle split names
    train_split = hf_ds['train']
    val_split = hf_ds['validation'] if 'validation' in hf_ds else hf_ds['test']
    
    print(f"  -> Train samples: {len(train_split):,}")
    print(f"  -> Val samples  : {len(val_split):,}")
    
    train_transform, val_transform = build_transforms(image_size)
    
    train_dataset = ImageNet100HFDataset(train_split, transform=train_transform)
    val_dataset = ImageNet100HFDataset(val_split, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False
    )
    
    return train_loader, val_loader

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Pre-download and verify ImageNet-100 dataset from Hugging Face Hub.")
    parser.add_argument('--hf_token', type=str, default=None, help="Hugging Face API token")
    parser.add_argument('--data_dir', type=str, default=None, help="Custom cache directory to save dataset")
    parser.add_argument('--batch_size', type=int, default=32, help="Test batch size")
    args = parser.parse_args()

    print("=" * 70)
    print("📥 Pre-downloading ImageNet-100 from Hugging Face Hub")
    print("=" * 70)
    train_loader, val_loader = get_imagenet100_loaders(
        batch_size=args.batch_size,
        hf_token=args.hf_token,
        data_dir=args.data_dir,
        num_workers=2
    )

    print("\nVerifying DataLoader batch retrieval...")
    x, y = next(iter(train_loader))
    print(f"  -> Sample Batch Image Tensor Shape : {x.shape} (Expected: [{args.batch_size}, 3, 224, 224])")
    print(f"  -> Sample Batch Target Tensor Shape: {y.shape}")
    print(f"  -> Label Range in Batch            : [{y.min().item()} .. {y.max().item()}]")
    print("=" * 70)
    print("✅ ImageNet-100 dataset downloaded and verified successfully!")
    print("   You can now start training with: python examples/vit_imagenet100/train.py")
    print("=" * 70)

