"""
Minimal script to load distilled ViT and extract features.
"""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from distilled_vit import DistilledViT

def load_distilled_vit(checkpoint_path):
    """Load distilled ViT from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    import ipdb; ipdb.set_trace()
    # Extract config
    config = checkpoint['config']
    teacher_model_name = checkpoint['teacher_model_name']
    
    # Infer teacher_embed_dim from checkpoint weights
    camera_weight_shape = checkpoint['model_state_dict']['teacher_projectors.camera_features.1.weight'].shape
    teacher_embed_dim = camera_weight_shape[1]  # Input dimension
    
    print(f"Detected teacher_embed_dim: {teacher_embed_dim}")
    
    # Create model with correct dimensions
    model = DistilledViT(
        teacher_embed_dim=teacher_embed_dim,
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Loaded DistilledViT from step {checkpoint['global_step']}")
    print(f"   Architecture: {config['EMBED_DIM']}d, {config['DEPTH']} layers, {config['NUM_HEADS']} heads")
    print(f"   Distill tokens: {config['DISTILL_TOKENS']}")
    
    return model

def preprocess_image(image_path):
    """Preprocess image for DistilledViT."""
    # ImageNet normalization (same as training)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def extract_features(model, image_tensor):
    """Extract features using distilled ViT."""
    with torch.no_grad():
        features = model(image_tensor)
    
    return features

# Example usage
if __name__ == "__main__":
    # Load model
    import ipdb; ipdb.set_trace()
    checkpoint_path = "checkpoints/distilled_vit_step_2000.pt"  # Update path
    model = load_distilled_vit(checkpoint_path)
    
    # Process image
    image_path = "rgb_frame_0.png"  # Update path
    image_tensor = preprocess_image(image_path)
    
    print(f"Image tensor shape: {image_tensor.shape}")
    
    # Extract features
    features = extract_features(model, image_tensor)
    
    print("\n📊 Extracted Features:")
    for feature_type, feature_tensor in features.items():
        print(f"  {feature_type}: {feature_tensor.shape}")
        print(f"    Mean: {feature_tensor.mean().item():.4f}")
        print(f"    Std:  {feature_tensor.std().item():.4f}")
    
    print("\n✅ Feature extraction complete!")