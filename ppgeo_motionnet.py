"""
PPGeo MotionNet for Stage 2 training.
Follows exact PPGeo implementation with ResNet34 encoder.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict

# Import PPGeo ResNet encoder
from ppgeo_resnet import PPGeoResnetEncoder


class MotionNet(nn.Module):
    """
    MotionNet for PPGeo Stage 2 training.
    Uses ResNet34 encoder exactly like original PPGeo implementation.
    """
    
    def __init__(self, resnet_layers: int = 34):
        super().__init__()
        
        # Use ResNet encoder exactly like original PPGeo
        self.visual_encoder = PPGeoResnetEncoder(
            num_layers=resnet_layers, 
            pretrained=True, 
            num_input_images=1  # Single image input
        )
        
        # Motion decoder follows original PoseDecoder architecture
        from collections import OrderedDict
        
        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.visual_encoder.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(1 * 256, 256, 3, 1, 1)  # num_input_features=1
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, 1, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * 2, 1)  # 6 params * 2 frames
        
        self.relu = nn.ReLU()
        self.net = nn.ModuleList(list(self.convs.values()))
    
    def forward(self, inputs: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for motion estimation - follows exact PPGeo implementation.
        Args:
            inputs: Dictionary with PPGeo-style tuple keys
        Returns:
            axisangle1, translation1, axisangle2, translation2
        """
        # Get frames from PPGeo format - exactly like original
        motion_inputs1 = inputs[("color_aug", -1, 0)]  # Previous frame
        motion_inputs2 = inputs[("color_aug", 0, 0)]   # Current frame
        
        # Encode both frames separately (like original PPGeo)
        enc1 = self.visual_encoder(motion_inputs1, normalize=True)
        enc2 = self.visual_encoder(motion_inputs2, normalize=True)
        
        # Apply pose decoder to each encoded frame
        axisangle1, translation1 = self._decode_pose([enc1])
        axisangle2, translation2 = self._decode_pose([enc2])
        
        return axisangle1, translation1, axisangle2, translation2
    
    def _decode_pose(self, input_features):
        """
        Decode pose from features - follows exact PPGeo PoseDecoder implementation.
        """
        # Extract last feature from each input
        last_features = [f[-1] for f in input_features]
        
        # Apply squeeze convolution and ReLU
        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)
        
        # Apply pose convolutions
        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)
        
        # Global average pooling
        out = out.mean(3).mean(2)
        
        # Reshape and scale by 0.01 (exactly like original)
        out = 0.01 * out.view(-1, 2, 1, 6)  # 2 frames to predict for
        
        axisangle = out[..., :3]
        translation = out[..., 3:]
        
        return axisangle, translation