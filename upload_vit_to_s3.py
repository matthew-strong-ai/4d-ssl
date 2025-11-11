"""
Upload distilled ViT checkpoint to S3.
"""

import os
from utils.s3_utils import upload_file_to_s3

def upload_vit_checkpoint(local_path, s3_bucket="research-datasets", s3_prefix="autonomy_checkpoints/distilled_vit"):
    """Upload ViT checkpoint to S3."""
    
    # Extract filename
    filename = os.path.basename(local_path)
    s3_key = f"{s3_prefix}/{filename}"
    s3_uri = f"s3://{s3_bucket}/{s3_key}"
    
    print(f"Uploading {local_path} to {s3_uri}")
    
    try:
        upload_file_to_s3(local_path, s3_uri)
        print(f"✅ Successfully uploaded to {s3_uri}")
        return s3_uri
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None

if __name__ == "__main__":
    # Upload specific checkpoint
    local_path = "checkpoints/distilled_vit_step_2000.pt"
    s3_uri = upload_vit_checkpoint(local_path)
    
    # Or upload all ViT checkpoints
    checkpoint_dir = "checkpoints"
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("distilled_vit_step_"):
            local_path = os.path.join(checkpoint_dir, filename)
            upload_vit_checkpoint(local_path)