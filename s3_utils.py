"""
S3 utilities for downloading and uploading files.
"""
import os
import boto3
from typing import Optional, Union
from pathlib import Path
import time


def get_s3_client():
    """Create and return an S3 client using environment variables."""
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
        region_name=os.getenv("REGION"),
        endpoint_url=os.getenv("ENDPOINT_URL")
    )


def download_file_from_s3(bucket: str, key: str, local_path: str, 
                         overwrite: bool = False, create_dirs: bool = True) -> bool:
    """
    Download a file from S3 to local filesystem.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key/path
        local_path: Local file path to save to
        overwrite: Whether to overwrite existing files
        create_dirs: Whether to create parent directories if they don't exist
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if file already exists
    if os.path.exists(local_path) and not overwrite:
        print(f"📄 File already exists: {local_path} (use overwrite=True to replace)")
        return True
    
    # Create parent directories if needed
    if create_dirs:
        parent_dir = Path(local_path).parent
        parent_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"📥 Downloading s3://{bucket}/{key} -> {local_path}")
        start_time = time.time()
        
        client = get_s3_client()
        client.download_file(bucket, key, local_path)
        
        end_time = time.time()
        file_size = os.path.getsize(local_path)
        print(f"✅ Downloaded {file_size:,} bytes in {end_time - start_time:.2f}s")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading s3://{bucket}/{key}: {e}")
        return False


def download_from_s3_uri(s3_uri: str, local_path: str, 
                        overwrite: bool = False, create_dirs: bool = True) -> bool:
    """
    Download a file from S3 using an s3:// URI.
    
    Args:
        s3_uri: S3 URI (e.g., 's3://bucket/path/to/file.txt')
        local_path: Local file path to save to
        overwrite: Whether to overwrite existing files
        create_dirs: Whether to create parent directories if they don't exist
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not s3_uri.startswith('s3://'):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    
    # Parse S3 URI
    s3_path = s3_uri[5:]  # Remove 's3://'
    bucket, key = s3_path.split('/', 1)
    
    return download_file_from_s3(bucket, key, local_path, overwrite, create_dirs)


def upload_file_to_s3(local_path: str, bucket: str, key: str, 
                     overwrite: bool = False) -> bool:
    """
    Upload a local file to S3.
    
    Args:
        local_path: Local file path to upload
        bucket: S3 bucket name
        key: S3 object key/path
        overwrite: Whether to overwrite existing S3 objects
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(local_path):
        print(f"❌ Local file does not exist: {local_path}")
        return False
    
    try:
        client = get_s3_client()
        
        # Check if object already exists
        if not overwrite:
            try:
                client.head_object(Bucket=bucket, Key=key)
                print(f"📄 Object already exists: s3://{bucket}/{key} (use overwrite=True to replace)")
                return True
            except client.exceptions.NoSuchKey:
                pass  # Object doesn't exist, proceed with upload
        
        print(f"📤 Uploading {local_path} -> s3://{bucket}/{key}")
        start_time = time.time()
        
        file_size = os.path.getsize(local_path)
        client.upload_file(local_path, bucket, key)
        
        end_time = time.time()
        print(f"✅ Uploaded {file_size:,} bytes in {end_time - start_time:.2f}s")
        return True
        
    except Exception as e:
        print(f"❌ Error uploading to s3://{bucket}/{key}: {e}")
        return False


def file_exists_in_s3(bucket: str, key: str) -> bool:
    """
    Check if a file exists in S3.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key/path
        
    Returns:
        bool: True if file exists, False otherwise
    """
    try:
        client = get_s3_client()
        client.head_object(Bucket=bucket, Key=key)
        return True
    except client.exceptions.NoSuchKey:
        return False
    except Exception as e:
        print(f"❌ Error checking s3://{bucket}/{key}: {e}")
        return False


def list_s3_objects(bucket: str, prefix: str = "", max_keys: int = 1000) -> list:
    """
    List objects in an S3 bucket with optional prefix filter.
    
    Args:
        bucket: S3 bucket name
        prefix: Object key prefix to filter by
        max_keys: Maximum number of objects to return
        
    Returns:
        list: List of object keys
    """
    try:
        client = get_s3_client()
        response = client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=max_keys
        )
        
        if 'Contents' in response:
            return [obj['Key'] for obj in response['Contents']]
        else:
            return []
            
    except Exception as e:
        print(f"❌ Error listing objects in s3://{bucket}/{prefix}: {e}")
        return []


# Example usage
if __name__ == "__main__":
    # Test downloading a file
    success = download_from_s3_uri(
        "s3://my-bucket/path/to/file.txt",
        "local_file.txt"
    )
    print(f"Download success: {success}")