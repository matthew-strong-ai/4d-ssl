"""
Upload PPGeo ResNet checkpoint to S3.
"""

import os
from utils.s3_utils import upload_file_to_s3

def upload_ppgeo_resnet_checkpoint(local_path, s3_bucket="research-datasets", s3_prefix="autonomy_checkpoints", wandb_run_name=None):
    """Upload PPGeo ResNet checkpoint to S3."""
    
    if not os.path.exists(local_path):
        print(f"❌ Checkpoint not found: {local_path}")
        return None
    
    print(f"📤 Uploading PPGeo ResNet checkpoint: {local_path}")
    
    try:
        success = upload_file_to_s3(
            file_path=local_path,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            wandb_run_name=wandb_run_name
        )
        
        if success:
            filename = os.path.basename(local_path)
            if wandb_run_name:
                name, ext = os.path.splitext(filename)
                filename = f"{wandb_run_name}_{name}{ext}"
            s3_uri = f"s3://{s3_bucket}/{s3_prefix}/{filename}"
            print(f"✅ Successfully uploaded to {s3_uri}")
            return s3_uri
        else:
            print(f"❌ Upload failed")
            return None
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None

if __name__ == "__main__":
    # Upload specific checkpoint
    local_path = "checkpoints/ppgeo_stage1_step_1100.pt"
    s3_uri = upload_ppgeo_resnet_checkpoint(local_path)
    
    # Or upload all ViT checkpoints
    # checkpoint_dir = "checkpoints"
    # for filename in os.listdir(checkpoint_dir):
    #     if filename.startswith("distilled_vit_step_"):
    #         local_path = os.path.join(checkpoint_dir, filename)
    #         upload_vit_checkpoint(local_path)