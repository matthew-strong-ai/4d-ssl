#!/bin/bash

# Example script for running Pi3 training with GSAM2 integration
# This demonstrates various GSAM2 configuration options

echo "🚀 Pi3 Training with GSAM2 Integration Examples"
echo "=============================================="

# Basic GSAM2 training - processes every 10 steps, saves detailed masks for first 5 steps only
echo "🔥 Example 1: Basic GSAM2 Training"
echo "python train_pi3.py \\"
echo "    --root_dir /path/to/your/images \\"
echo "    --use_gsam2 \\"
echo "    --gsam2_frequency 10 \\"
echo "    --gsam2_save_masks \\"
echo "    --gsam2_save_masks_max_steps 5 \\"
echo "    --num_epochs 5 \\"
echo "    --learning_rate 1e-4"
echo ""

# Custom prompt training - detects specific objects
echo "🎯 Example 2: Custom Prompt Training"
echo "python train_pi3.py \\"
echo "    --root_dir /path/to/your/images \\"
echo "    --use_gsam2 \\"
echo "    --gsam2_prompt \"pedestrian. cyclist. car. truck.\" \\"
echo "    --gsam2_box_threshold 0.3 \\"
echo "    --gsam2_text_threshold 0.4 \\"
echo "    --gsam2_frequency 5 \\"
echo "    --gsam2_save_masks \\"
echo "    --gsam2_save_dir \"custom_detections\""
echo ""

# High-frequency processing - every step
echo "🔍 Example 3: High-Frequency Processing"
echo "python train_pi3.py \\"
echo "    --root_dir /path/to/your/images \\"
echo "    --use_gsam2 \\"
echo "    --gsam2_frequency 0 \\"
echo "    --gsam2_prompt \"person. vehicle.\" \\"
echo "    --num_epochs 2"
echo ""

# Training without GSAM2 - normal Pi3 training
echo "🚫 Example 4: Normal Training (GSAM2 disabled)"
echo "python train_pi3.py \\"
echo "    --root_dir /path/to/your/images \\"
echo "    --num_epochs 10 \\"
echo "    --learning_rate 2e-5"
echo ""

echo "💡 Tips:"
echo "- Set --gsam2_frequency 0 to process every step (slower but comprehensive)"
echo "- Set --gsam2_frequency 10+ for periodic sampling (faster training)"
echo "- Use --gsam2_save_masks to save detailed PNG visualizations"
echo "- Set --gsam2_save_masks_max_steps to limit visualization saving (e.g., 5 for first 5 steps only)"
echo "- Set --gsam2_save_masks_max_steps 0 to save visualizations for all steps"
echo "- Adjust thresholds based on your detection quality needs"
echo "- GSAM2 runs with torch.no_grad() - no impact on training gradients"
echo ""

echo "📁 Output Structure (when --gsam2_save_masks enabled):"
echo "gsam2_masks/"
echo "├── step_000010/"
echo "│   ├── masks.npz                    # Raw mask data"
echo "│   ├── frame_00_original.png        # Original frames" 
echo "│   ├── frame_00_mask_obj1.png       # Individual masks"
echo "│   ├── frame_00_overlay.png         # Colored overlays"
echo "│   ├── frame_00_comparison.png      # Side-by-side view"
echo "│   ├── frame_00_bboxes.png          # Bounding boxes"
echo "│   ├── sequence_summary.png         # Multi-frame overview"
echo "│   └── summary.txt                  # Detection summary"
echo "├── step_000020/"
echo "│   └── ..."
echo ""

echo "🎬 WandB Integration:"
echo "- GSAM2 metrics automatically logged to WandB"
echo "- Track: num_objects, masks_per_frame, detection_scores" 
echo "- Runs tagged with 'gsam2' when enabled"
echo ""

echo "Ready to run! Replace '/path/to/your/images' with your actual image directory."