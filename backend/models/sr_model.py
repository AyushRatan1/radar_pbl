"""
Super-Resolution Model for SAR Image Enhancement
Uses ESPCN (Efficient Sub-Pixel Convolutional Neural Network) architecture
Optimized for automotive radar/SAR imagery
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io


class ESPCN(nn.Module):
    """
    Efficient Sub-Pixel Convolutional Neural Network for Super-Resolution
    A lightweight CNN-based model perfect for real-time automotive applications
    """
    
    def __init__(self, scale_factor=4, num_channels=1):
        super(ESPCN, self).__init__()
        self.scale_factor = scale_factor
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        
        # Sub-pixel convolution for upscaling
        self.conv4 = nn.Conv2d(32, num_channels * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x


class SARSuperResolution:
    """
    SAR Image Super-Resolution Handler
    Manages model loading, inference, and image processing
    """
    
    def __init__(self, scale_factor=4):
        self.scale_factor = scale_factor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ESPCN(scale_factor=scale_factor, num_channels=1)
        self.model.to(self.device)
        self.model.eval()
        
        # Try to load trained weights
        self.weights_loaded = False
        weights_paths = [
            'weights/espcn_best.pth',
            'weights/espcn_final.pth',
        ]
        
        for weights_path in weights_paths:
            if os.path.exists(weights_path):
                try:
                    self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                    self.weights_loaded = True
                    print(f"Loaded trained weights from {weights_path}")
                    break
                except Exception as e:
                    print(f"Failed to load weights from {weights_path}: {e}")
        
        if not self.weights_loaded:
            print("Using randomly initialized weights. Run 'python train.py' to train the model.")
        
        print(f"SAR Super-Resolution Model initialized on {self.device}")
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor for model input"""
        # Convert to grayscale (SAR images are typically single-channel)
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert to numpy and normalize
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
    
    def postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """Convert model output tensor back to PIL Image"""
        # Remove batch and channel dimensions
        output = tensor.squeeze().cpu().detach().numpy()
        
        # Clip values and convert to uint8
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        
        return Image.fromarray(output, mode='L')
    
    def enhance(self, image: Image.Image) -> Image.Image:
        """
        Enhance a SAR image using super-resolution
        
        Args:
            image: Input PIL Image
            
        Returns:
            Enhanced PIL Image with 4x resolution
        """
        with torch.no_grad():
            # Preprocess
            input_tensor = self.preprocess(image)
            
            # Run inference
            output_tensor = self.model(input_tensor)
            
            # Postprocess
            enhanced_image = self.postprocess(output_tensor)
            
        return enhanced_image
    
    def enhance_with_comparison(self, image: Image.Image) -> tuple:
        """
        Enhance image and return both original (upscaled) and enhanced versions
        
        Returns:
            Tuple of (upscaled_original, enhanced) PIL Images
        """
        # Get enhanced version
        enhanced = self.enhance(image)
        
        # Upscale original using bicubic for comparison
        original_size = image.size
        new_size = (original_size[0] * self.scale_factor, original_size[1] * self.scale_factor)
        
        if image.mode != 'L':
            image = image.convert('L')
        upscaled_original = image.resize(new_size, Image.BICUBIC)
        
        return upscaled_original, enhanced


# Alternative: SRCNN Model (even simpler)
class SRCNN(nn.Module):
    """
    Super-Resolution Convolutional Neural Network
    The classic CNN for image super-resolution
    """
    
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x
