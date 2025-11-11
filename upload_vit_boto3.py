"""
Upload distilled ViT checkpoint to S3 using boto3.
"""

import boto3
import os
from botocore.exceptions import NoCredentialsError

def upload_to_s3_boto3(local_file, bucket, s3_file):
    """Upload file to S3 using boto3."""
    s3 = boto3.client('s3')
    
    try:
        s3.upload_file(local_file, bucket, s3_file)
        print(f"✅ Upload successful: s3://{bucket}/{s3_file}")
        return True
    except FileNotFoundError:
        print(f"❌ File {local_file} not found")
        return False
    except NoCredentialsError:
        print("❌ Credentials not available")
        return False
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        return False

if __name__ == "__main__":
    # Configuration
    local_file = "checkpoints/distilled_vit_step_2000.pt"
    bucket = "research-datasets"
    s3_file = "autonomy_checkpoints/distilled_vit/distilled_vit_step_2000.pt"
    
    # Upload
    upload_to_s3_boto3(local_file, bucket, s3_file)