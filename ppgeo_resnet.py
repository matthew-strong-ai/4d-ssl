"""
PPGeo ResNet implementation following the original PPGeo architecture.
Based on the original PPGeo code from /home/matthew_strong/prayers/PPGeo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import numpy as np
from typing import List, Dict, Tuple
from collections import OrderedDict


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from the original PPGeo implementation.
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        # Use modern torchvision approach
        if num_layers == 18:
            pretrained_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif num_layers == 50:
            pretrained_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Get state dict and modify conv1 for multi-image input
        loaded = pretrained_model.state_dict()
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class PPGeoResnetEncoder(nn.Module):
    """ResNet encoder for PPGeo depth estimation following original implementation."""
    
    def __init__(self, num_layers=18, pretrained=True, num_input_images=1):
        super(PPGeoResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image, normalize=False):
        """
        Extract multi-scale features from ResNet.
        Args:
            input_image: Input images [B, 3, H, W] or [B, 6, H, W] for pose
            normalize: Whether to apply ImageNet normalization
        Returns:
            List of features at different scales
        """
        features = []
        
        if normalize:
            # ImageNet normalization
            std = torch.tensor([0.229, 0.224, 0.225]).type_as(input_image).view(1, 3, 1, 1)
            mean = torch.tensor([0.485, 0.456, 0.406]).type_as(input_image).view(1, 3, 1, 1)
            
            # Handle multi-image input
            if input_image.shape[1] > 3:
                std = std.repeat(1, input_image.shape[1]//3, 1, 1)
                mean = mean.repeat(1, input_image.shape[1]//3, 1, 1)
            
            x = (input_image - mean) / std
        else:
            x = input_image
            
        # Extract features at multiple scales
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        features.append(self.encoder.relu(x))  # Scale 0: [B, 64, H/2, W/2]
        features.append(self.encoder.layer1(self.encoder.maxpool(features[-1])))  # Scale 1: [B, 64, H/4, W/4]
        features.append(self.encoder.layer2(features[-1]))  # Scale 2: [B, 128, H/8, W/8]
        features.append(self.encoder.layer3(features[-1]))  # Scale 3: [B, 256, H/16, W/16]
        features.append(self.encoder.layer4(features[-1]))  # Scale 4: [B, 512, H/32, W/32]

        return features


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input"""
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


def upsample(x):
    """Upsample input tensor by a factor of 2"""
    return F.interpolate(x, scale_factor=2, mode="nearest")


class PPGeoResnetDepthDecoder(nn.Module):
    """ResNet depth decoder following original PPGeo implementation."""
    
    def __init__(self, num_ch_enc, scales=[0, 1, 2, 3], num_output_channels=1, use_skips=True):
        super(PPGeoResnetDepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features) -> Dict[str, torch.Tensor]:
        """
        Generate depth predictions using ResNet decoder.
        Args:
            input_features: List of features from ResNet encoder
        Returns:
            Dictionary of depth predictions at multiple scales
        """
        outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return outputs


class PPGeoResnetPoseDecoder(nn.Module):
    """ResNet pose decoder following original PPGeo implementation."""
    
    def __init__(self, num_ch_enc, num_input_features=1, num_frames_to_predict_for=2):
        super(PPGeoResnetPoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        self.num_frames_to_predict_for = num_frames_to_predict_for

        # Create modules properly
        self.squeeze = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.pose_conv_0 = nn.Conv2d(num_input_features * 256, 256, 3, 1, 1)
        self.pose_conv_1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.pose_conv_2 = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

    def forward(self, input_features):
        """
        Predict camera pose from ResNet features.
        Args:
            input_features: List of features from ResNet encoder (only uses last level)
        Returns:
            axisangle, translation
        """
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.squeeze(f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        out = self.pose_conv_0(out)
        out = self.relu(out)
        out = self.pose_conv_1(out)
        out = self.relu(out)
        out = self.pose_conv_2(out)

        out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation


class PPGeoResnetPoseCNN(nn.Module):
    """Alternative pose network using CNN architecture from original PPGeo."""
    
    def __init__(self, num_input_frames):
        super(PPGeoResnetPoseCNN, self).__init__()

        self.num_input_frames = num_input_frames

        self.convs = {}
        self.convs[0] = nn.Conv2d(3 * num_input_frames, 16, 7, 2, 3)
        self.convs[1] = nn.Conv2d(16, 32, 5, 2, 2)
        self.convs[2] = nn.Conv2d(32, 64, 3, 2, 1)
        self.convs[3] = nn.Conv2d(64, 128, 3, 2, 1)
        self.convs[4] = nn.Conv2d(128, 256, 3, 2, 1)
        self.convs[5] = nn.Conv2d(256, 256, 3, 2, 1)
        self.convs[6] = nn.Conv2d(256, 256, 3, 2, 1)

        self.pose_conv = nn.Conv2d(256, 6 * (num_input_frames - 1), 1)

        self.num_convs = len(self.convs)
        self.relu = nn.ReLU(True)
        self.net = nn.ModuleList(list(self.convs.values()))

        # Intrinsics prediction heads
        self.fl = nn.Sequential(nn.Linear(256, 128),
                                nn.ReLU(True),
                                nn.Linear(128, 2),
                                nn.Softplus())

        self.offset = nn.Sequential(nn.Linear(256, 128),
                                    nn.ReLU(True),
                                    nn.Linear(128, 2),
                                    nn.Sigmoid())

        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input_images):
        """
        Forward pass through pose CNN.
        Args:
            input_images: Concatenated images [B, 3*num_frames, H, W]
        Returns:
            axisangle, translation, focal_length, offset
        """
        out = input_images
        
        for i in range(self.num_convs):
            out = self.convs[i](out)
            out = self.relu(out)

        # Intrinsics prediction
        feature_pooled = torch.flatten(self.avg_pooling(out), 1)
        fl = self.fl(feature_pooled)
        offset = self.offset(feature_pooled)

        # Pose prediction
        pose_out = self.pose_conv(out)
        pose_out = pose_out.mean(3).mean(2)
        pose_out = 0.01 * pose_out.view(-1, self.num_input_frames - 1, 1, 6)

        axisangle = pose_out[..., :3]
        translation = pose_out[..., 3:]

        return axisangle, translation, fl, offset