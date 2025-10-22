#!/bin/bash

# Installation script for PyTorch with CUDA 12.1 support
echo "🚀 Installing PyTorch with CUDA 12.1 support..."

# Install PyTorch with CUDA 12.1
pip install torch>=2.1.0 torchvision>=0.16.0 torchaudio>=2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
echo "📦 Installing other dependencies..."
pip install -r requirements_base.txt

echo "✅ Installation complete!"
echo "🔍 Verifying CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'PyTorch version: {torch.__version__}')"