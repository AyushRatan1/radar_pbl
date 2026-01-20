"""
ESPCN Model Training Script for SAR Image Super-Resolution

This script:
1. Downloads/generates training data for SAR-like images
2. Trains the ESPCN model
3. Saves trained weights

Usage:
    python train.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import urllib.request
import zipfile
import shutil
from pathlib import Path

from models.sr_model import ESPCN


# Configuration
CONFIG = {
    'scale_factor': 4,
    'batch_size': 16,
    'epochs': 100,
    'learning_rate': 0.001,
    'patch_size': 64,  # Size of training patches
    'data_dir': 'data',
    'weights_dir': 'weights',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


class SARDataset(Dataset):
    """
    Dataset for SAR image super-resolution training.
    Creates low-resolution / high-resolution pairs from images.
    """
    
    def __init__(self, image_dir, patch_size=64, scale_factor=4, augment=True):
        self.image_dir = Path(image_dir)
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.augment = augment
        
        # Find all images
        self.image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
            self.image_paths.extend(self.image_dir.glob(ext))
        
        print(f"Found {len(self.image_paths)} training images")
        
        # Transforms
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.image_paths) * 10  # Multiple patches per image
    
    def __getitem__(self, idx):
        # Get image
        img_idx = idx % len(self.image_paths)
        img_path = self.image_paths[img_idx]
        
        # Load and convert to grayscale
        img = Image.open(img_path).convert('L')
        
        # Random crop for high-res patch
        hr_size = self.patch_size * self.scale_factor
        
        # Ensure image is large enough
        if img.width < hr_size or img.height < hr_size:
            img = img.resize((hr_size, hr_size), Image.BICUBIC)
        
        # Random crop
        left = np.random.randint(0, max(1, img.width - hr_size))
        top = np.random.randint(0, max(1, img.height - hr_size))
        hr_patch = img.crop((left, top, left + hr_size, top + hr_size))
        
        # Augmentation
        if self.augment:
            if np.random.random() > 0.5:
                hr_patch = hr_patch.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.random() > 0.5:
                hr_patch = hr_patch.transpose(Image.FLIP_TOP_BOTTOM)
            if np.random.random() > 0.5:
                hr_patch = hr_patch.rotate(90)
        
        # Create low-res version
        lr_patch = hr_patch.resize(
            (self.patch_size, self.patch_size), 
            Image.BICUBIC
        )
        
        # Convert to tensors
        hr_tensor = self.to_tensor(hr_patch)
        lr_tensor = self.to_tensor(lr_patch)
        
        return lr_tensor, hr_tensor


def download_training_data():
    """
    Download sample training data.
    Uses BSD100/BSD200 dataset as base, then synthesizes SAR-like characteristics.
    """
    data_dir = Path(CONFIG['data_dir'])
    data_dir.mkdir(exist_ok=True)
    
    train_dir = data_dir / 'train'
    if train_dir.exists() and len(list(train_dir.glob('*'))) > 0:
        print(f"Training data already exists in {train_dir}")
        return train_dir
    
    train_dir.mkdir(exist_ok=True)
    
    print("Downloading training dataset...")
    print("Note: Using DIV2K-like synthetic images for training")
    
    # Generate synthetic SAR-like images for training
    print("Generating synthetic SAR training images...")
    generate_synthetic_sar_images(train_dir, num_images=200)
    
    return train_dir


def generate_synthetic_sar_images(output_dir, num_images=200):
    """
    Generate synthetic SAR-like images for training.
    SAR images typically have:
    - Grayscale
    - Speckle noise
    - High contrast features
    - Geometric patterns
    """
    output_dir = Path(output_dir)
    
    for i in range(num_images):
        # Random size between 256 and 512
        size = np.random.randint(256, 512)
        
        # Create base image with random patterns
        img = np.zeros((size, size), dtype=np.float32)
        
        # Add random geometric shapes (simulating radar returns)
        num_shapes = np.random.randint(5, 20)
        for _ in range(num_shapes):
            shape_type = np.random.choice(['rect', 'circle', 'line'])
            intensity = np.random.uniform(0.5, 1.0)
            
            if shape_type == 'rect':
                x1 = np.random.randint(0, size - 20)
                y1 = np.random.randint(0, size - 20)
                w = np.random.randint(10, 50)
                h = np.random.randint(10, 50)
                img[y1:y1+h, x1:x1+w] = intensity
                
            elif shape_type == 'circle':
                cx, cy = np.random.randint(20, size-20, 2)
                r = np.random.randint(5, 30)
                y, x = np.ogrid[:size, :size]
                mask = (x - cx)**2 + (y - cy)**2 <= r**2
                img[mask] = intensity
                
            elif shape_type == 'line':
                x1, y1 = np.random.randint(0, size, 2)
                x2, y2 = np.random.randint(0, size, 2)
                # Simple line drawing
                num_points = max(abs(x2-x1), abs(y2-y1))
                if num_points > 0:
                    for t in np.linspace(0, 1, num_points):
                        x = int(x1 + t * (x2 - x1))
                        y = int(y1 + t * (y2 - y1))
                        if 0 <= x < size and 0 <= y < size:
                            img[y, x] = intensity
        
        # Add background texture
        noise = np.random.uniform(0, 0.3, (size, size))
        img = img + noise
        
        # Add speckle noise (multiplicative, characteristic of SAR)
        speckle = np.random.exponential(1.0, (size, size))
        img = img * speckle
        
        # Normalize to 0-255
        img = np.clip(img, 0, None)
        img = (img / img.max() * 255).astype(np.uint8)
        
        # Add some Gaussian smoothing
        from PIL import ImageFilter
        pil_img = Image.fromarray(img, mode='L')
        if np.random.random() > 0.5:
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1))
        
        # Save
        pil_img.save(output_dir / f'sar_synthetic_{i:04d}.png')
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{num_images} images")
    
    print(f"Generated {num_images} synthetic SAR images in {output_dir}")


def train_model():
    """Main training loop"""
    
    print("="*60)
    print("ESPCN Training for SAR Super-Resolution")
    print("="*60)
    print(f"Device: {CONFIG['device']}")
    print(f"Scale Factor: {CONFIG['scale_factor']}x")
    print(f"Batch Size: {CONFIG['batch_size']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print("="*60)
    
    # Prepare data
    train_dir = download_training_data()
    
    # Create dataset and dataloader
    dataset = SARDataset(
        train_dir,
        patch_size=CONFIG['patch_size'],
        scale_factor=CONFIG['scale_factor'],
        augment=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )
    
    # Initialize model
    model = ESPCN(scale_factor=CONFIG['scale_factor'], num_channels=1)
    model = model.to(CONFIG['device'])
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Training loop
    best_loss = float('inf')
    weights_dir = Path(CONFIG['weights_dir'])
    weights_dir.mkdir(exist_ok=True)
    
    print("\nStarting training...")
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(dataloader):
            lr_imgs = lr_imgs.to(CONFIG['device'])
            hr_imgs = hr_imgs.to(CONFIG['device'])
            
            # Forward pass
            optimizer.zero_grad()
            sr_imgs = model(lr_imgs)
            
            # Compute loss
            loss = criterion(sr_imgs, hr_imgs)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / num_batches
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] - Loss: {avg_loss:.6f} - LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), weights_dir / 'espcn_best.pth')
    
    # Save final model
    torch.save(model.state_dict(), weights_dir / 'espcn_final.pth')
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best Loss: {best_loss:.6f}")
    print(f"Weights saved to: {weights_dir}")
    print("="*60)
    
    return model


def test_model():
    """Quick test of the trained model"""
    weights_path = Path(CONFIG['weights_dir']) / 'espcn_best.pth'
    
    if not weights_path.exists():
        print("No trained weights found. Run training first.")
        return
    
    print("\nTesting trained model...")
    
    model = ESPCN(scale_factor=CONFIG['scale_factor'], num_channels=1)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    
    # Create a test image
    test_img = np.random.rand(64, 64).astype(np.float32)
    test_tensor = torch.from_numpy(test_img).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        output = model(test_tensor)
    
    expected_size = 64 * CONFIG['scale_factor']
    actual_size = output.shape[-1]
    
    print(f"Input size: 64x64")
    print(f"Output size: {actual_size}x{actual_size}")
    print(f"Expected: {expected_size}x{expected_size}")
    print(f"Test: {'PASSED' if actual_size == expected_size else 'FAILED'}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ESPCN for SAR Super-Resolution')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--test-only', action='store_true', help='Only test the model')
    
    args = parser.parse_args()
    
    CONFIG['epochs'] = args.epochs
    CONFIG['batch_size'] = args.batch_size
    CONFIG['learning_rate'] = args.lr
    
    if args.test_only:
        test_model()
    else:
        train_model()
        test_model()
