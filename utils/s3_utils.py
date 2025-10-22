#!/usr/bin/env python3
"""
S3 utilities for checkpoint and file management.

This module contains functions for uploading and managing files in S3 buckets,
specifically optimized for training checkpoint management.
"""

import os
import subprocess
from io import BytesIO
import boto3
import torch


def save_state_dict_to_s3(state_dict, s3_path: str):
    """
    Save a PyTorch state dict to an S3 bucket.
    
    Args:
        state_dict: The PyTorch state dict to save
        s3_path (str): The S3 path (e.g., 's3://bucket/key.pth')
    """
    assert s3_path.startswith("s3://"), "Not a valid S3 path"
    s3_path = s3_path[5:]  # remove 's3://'
    bucket, key = s3_path.split("/", 1)
    
    print(f"📤 Starting state dict upload to s3://{bucket}/{key}")
    
    # Serialize the state dict
    buffer = BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    
    # Upload to S3
    session = boto3.Session(
        aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
        region_name=os.getenv("REGION"),
    )
    client = session.client("s3", endpoint_url=os.getenv("ENDPOINT_URL"))
    client.upload_fileobj(buffer, bucket, key)
    print(f"✅ Successfully saved state dict to s3://{bucket}/{key}")


def upload_file_to_s3(file_path, s3_bucket="research-datasets", s3_prefix="autonomy_checkpoints", wandb_run_name=None):
    """
    Upload any file to S3 using AWS CLI.
    
    Args:
        file_path: Local path to file
        s3_bucket: S3 bucket name
        s3_prefix: S3 prefix/folder
        wandb_run_name: WandB run name to include in filename
        
    Returns:
        bool: True if upload successful, False otherwise
    """
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        return False
    
    # Get base filename and modify it to include wandb run name
    base_filename = os.path.basename(file_path)
    
    if wandb_run_name:
        # Split filename and extension
        name, ext = os.path.splitext(base_filename)
        # Create new filename with wandb run name
        filename = f"{wandb_run_name}_{name}{ext}"
    else:
        filename = base_filename
    
    s3_path = f"s3://{s3_bucket}/{s3_prefix}/{filename}"
    
    try:
        print(f"📤 Uploading file to S3: {s3_path}")
        
        # Use AWS CLI to upload
        cmd = ["aws", "s3", "cp", file_path, s3_path]

        # Set AWS environment variables for checksum validation
        env = os.environ.copy()
        env["AWS_REQUEST_CHECKSUM_CALCULATION"] = "when_required"
        env["AWS_RESPONSE_CHECKSUM_VALIDATION"] = "when_required"
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)  # 5 min timeout
        
        if result.returncode == 0:
            print(f"✅ Successfully uploaded file to S3: {s3_path}")
            return True
        else:
            print(f"❌ Failed to upload file to S3:")
            print(f"   Command: {' '.join(cmd)}")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ S3 upload timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error uploading checkpoint to S3: {e}")
        return False


def load_state_dict_from_s3(s3_path: str):
    """
    Load a PyTorch state dict from an S3 bucket.
    
    Args:
        s3_path (str): The S3 path (e.g., 's3://bucket/key.pth')
        
    Returns:
        dict: The loaded state dict
    """
    assert s3_path.startswith("s3://"), "Not a valid S3 path"
    s3_path = s3_path[5:]  # remove 's3://'
    bucket, key = s3_path.split("/", 1)
    
    print(f"📥 Loading state dict from s3://{bucket}/{key}")
    
    # Download from S3
    session = boto3.Session(
        aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
        region_name=os.getenv("REGION"),
    )
    client = session.client("s3", endpoint_url=os.getenv("ENDPOINT_URL"))
    
    buffer = BytesIO()
    client.download_fileobj(bucket, key, buffer)
    buffer.seek(0)
    
    # Load the state dict
    state_dict = torch.load(buffer, map_location='cpu')
    print(f"✅ Successfully loaded state dict from s3://{bucket}/{key}")
    
    return state_dict


def s3_file_exists(s3_path: str):
    """
    Check if a file exists in S3.
    
    Args:
        s3_path (str): The S3 path (e.g., 's3://bucket/key.pth')
        
    Returns:
        bool: True if file exists, False otherwise
    """
    assert s3_path.startswith("s3://"), "Not a valid S3 path"
    s3_path = s3_path[5:]  # remove 's3://'
    bucket, key = s3_path.split("/", 1)
    
    try:
        session = boto3.Session(
            aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
            region_name=os.getenv("REGION"),
        )
        client = session.client("s3", endpoint_url=os.getenv("ENDPOINT_URL"))
        client.head_object(Bucket=bucket, Key=key)
        return True
    except:
        return False