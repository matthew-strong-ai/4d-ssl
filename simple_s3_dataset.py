import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import io
from typing import List, Optional, Tuple
import re


class S3Dataset(Dataset):
    """
    Simple S3 dataset for consecutive images.
    
    Reads consecutive image sequences from S3 buckets for training.
    Uses PyTorch DataLoader's built-in prefetch_factor for efficient loading.
    
    Each sample returns (current_frames, future_frames) where:
    - current_frames: [m, C, H, W] tensor of input frames
    - future_frames: [n, C, H, W] tensor of target frames
    
    Expected S3 structure:
    bucket/
    ├── sequence1/
    │   ├── image_000.jpg
    │   ├── image_001.jpg
    │   └── ...
    ├── sequence2/
    │   ├── image_000.jpg
    │   ├── image_001.jpg
    │   └── ...
    └── ...
    """
    
    def __init__(
        self,
        bucket_name: str,
        sequence_prefixes: List[str],
        m: int = 3,
        n: int = 3,
        image_extension: str = ".jpg",
        transform: Optional[T.Compose] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = "us-east-1",
        preload_bytes: bool = False  # New parameter for bytes pre-caching
    ):
        """
        Args:
            bucket_name: S3 bucket name
            sequence_prefixes: List of sequence prefixes (e.g., ["sequence1/", "sequence2/"])
            m: Number of input frames
            n: Number of target frames  
            image_extension: Image file extension
            transform: Optional transforms to apply to images
            aws_access_key_id: AWS access key (if None, uses default credentials)
            aws_secret_access_key: AWS secret key (if None, uses default credentials)
            region_name: AWS region
            preload_bytes: If True, pre-download all image bytes for multiprocessing compatibility
        """
        self.bucket_name = bucket_name
        self.sequence_prefixes = sequence_prefixes
        self.m = m
        self.n = n
        self.total_frames = m + n
        self.image_extension = image_extension
        self.image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")  # Add supported extensions
        self.transform = transform or self._default_transform()
        self.preload_bytes = preload_bytes
        
        # Get AWS credentials from environment variables or parameters
        aws_access_key_id = aws_access_key_id or os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = aws_secret_access_key or os.environ.get('AWS_SECRET_ACCESS_KEY')
        region_name = "us-phoenix-1"
        endpoint_url = os.environ.get('AWS_ENDPOINT_URL')  # For S3-compatible services like MinIO
        
        # Debug credential information (without exposing secrets)
        print(f"🔐 Credential Debug Info:")
        print(f"   Access Key ID: {'✓ Set' if aws_access_key_id else '✗ Missing'}")
        print(f"   Secret Key: {'✓ Set' if aws_secret_access_key else '✗ Missing'}")
        print(f"   Region: {region_name}")
        print(f"   Endpoint URL: {endpoint_url if endpoint_url else 'Default (AWS S3)'}")
        
        # Initialize S3 client
        try:
            client_kwargs = {
                'region_name': region_name
            }
            
            # Add credentials if provided
            if aws_access_key_id and aws_secret_access_key:
                client_kwargs.update({
                    'aws_access_key_id': aws_access_key_id,
                    'aws_secret_access_key': aws_secret_access_key,
                })
                print(f"   Using explicit credentials")
            else:
                print(f"   Using default AWS credential chain")
                # For non-AWS S3 services, explicit credentials are usually required
                if endpoint_url:
                    print(f"   WARNING: Custom endpoint detected but no explicit credentials provided!")
            
            # Add custom endpoint if specified (for S3-compatible services)
            print(client_kwargs)
            if endpoint_url:
                client_kwargs['endpoint_url'] = endpoint_url
                # For S3-compatible services, we might need additional config
                client_kwargs['config'] = boto3.session.Config(
                    signature_version='s3v4',
                    s3={
                        'addressing_style': 'path'  # Use path-style addressing
                    }
                )
                
            self.s3_client = boto3.client('s3', **client_kwargs)
            print(f"✓ S3 client initialized successfully")
            
        except NoCredentialsError as e:
            raise ValueError(
                f"AWS credentials not found: {e}. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
                "environment variables, configure AWS CLI, or provide credentials explicitly."
            )
        except Exception as e:
            print(f"✗ Error initializing S3 client: {e}")
            raise
        
        # Build dataset index
        print(f"Building dataset index from S3 bucket '{bucket_name}'...")
        self.samples = self._build_dataset_index()
        print(f"Found {len(self.samples)} consecutive image sequences.")
        
        if len(self.samples) == 0:
            raise ValueError("No valid consecutive sequences found. Check bucket contents and sequence_prefixes.")
        
        # Initialize byte cache
        self.bytes_cache = {}  # {s3_key: bytes}
        
        # Optionally preload all bytes for multiprocessing compatibility
        if self.preload_bytes:
            print(f"🔄 Preloading image bytes for multiprocessing compatibility...")
            self._preload_all_bytes()
            print(f"✅ Preloaded {len(self.bytes_cache)} image files as bytes")
        
    
    def _default_transform(self) -> T.Compose:
        """Default image transforms"""
        return T.Compose([
            T.ToTensor(),
        ])
    
    def _natural_sort(self, key_list: List[str]) -> List[str]:
        """
        Sort S3 keys naturally based on filename.
        """
        def natural_key(s3_key):
            filename = os.path.basename(s3_key)
            return [int(text) if text.isdigit() else text.lower() for text in re.split('(\\d+)', filename)]
        
        return sorted(key_list, key=natural_key)
    
    def _preload_all_bytes(self):
        """Pre-download all unique image bytes for multiprocessing compatibility"""
        unique_keys = set()
        
        # Collect all unique image keys
        for sequence_prefix, start_idx, image_keys in self.samples:
            consecutive_keys = image_keys[start_idx:start_idx + self.total_frames]
            unique_keys.update(consecutive_keys)
        
        print(f"   Downloading {len(unique_keys)} unique images...")
        
        # Download all bytes
        for i, key in enumerate(unique_keys):
            if i % 100 == 0:
                print(f"   Progress: {i}/{len(unique_keys)} images downloaded")
            
            try:
                self.bytes_cache[key] = self._get_image_bytes_from_s3(key)
            except Exception as e:
                print(f"   Warning: Failed to download {key}: {e}")
                self.bytes_cache[key] = b''  # Empty bytes as fallback
    
    def _build_dataset_index(self) -> List[Tuple[str, int, List[str]]]:
        """
        Build index of valid consecutive sequences using paginated S3 listing.
        Returns list of (sequence_prefix, start_index, image_keys) tuples.
        """
        samples = []
        for sequence_prefix in self.sequence_prefixes:
            try:
                # Get all image keys for this sequence using paginator
                image_keys = []
                paginator = self.s3_client.get_paginator('list_objects_v2')
                
                print(f"📂 Listing objects for prefix: {sequence_prefix}")
                
                # List all objects with the given prefix using pagination
                for page in paginator.paginate(Bucket=self.bucket_name, Prefix=sequence_prefix):
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            key = obj['Key']
                            # Extract filename from key
                            filename = os.path.basename(key)
                            if filename.lower().endswith(self.image_extensions):
                                image_keys.append(key)
                
                if not image_keys:
                    print(f"⚠️  Warning: No objects found for prefix '{sequence_prefix}'")
                    continue
                
                # Sort files naturally (numerical ordering)
                image_keys = self._natural_sort(image_keys)
                
                # Skip first 300 frames and use the rest
                skip_first_frames = 300
                if len(image_keys) > skip_first_frames:
                    print(f"⏭️ Skipping first {skip_first_frames} frames in sequence '{sequence_prefix}' (had {len(image_keys)} total)")
                    image_keys = image_keys[skip_first_frames:]  # Skip first 300, use 300+
                else:
                    print(f"⚠️  Warning: Sequence '{sequence_prefix}' has only {len(image_keys)} frames, cannot skip {skip_first_frames} frames")
                    # Keep all frames if we don't have enough to skip
                
                # Create samples for consecutive sequences
                num_images = len(image_keys)
                if num_images < self.total_frames:
                    print(f"Warning: Sequence '{sequence_prefix}' has only {num_images} images, need {self.total_frames}")
                    continue
                
                # Generate all possible consecutive sequences
                for start_idx in range(num_images - self.total_frames + 1):
                    samples.append((sequence_prefix, start_idx, image_keys))
                
                print(f"✓ Sequence '{sequence_prefix}': {num_images} images -> {len(range(num_images - self.total_frames + 1))} samples")
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                error_message = e.response['Error']['Message']
                print(f"❌ Error accessing sequence '{sequence_prefix}': {error_code} - {error_message}")
                
                # Provide specific guidance based on error type
                if error_code == 'SignatureDoesNotMatch':
                    print("   🔧 Fix: This usually means:")
                    print("      • Secret key is incorrect or missing")
                    print("      • For Oracle Cloud/non-AWS: Make sure region matches your tenancy region")
                    print("      • Check that AWS_SECRET_ACCESS_KEY is set correctly")
                elif error_code == 'InvalidAccessKeyId':
                    print("   🔧 Fix: Access Key ID is invalid or missing")
                    print("      • Check AWS_ACCESS_KEY_ID environment variable")
                elif error_code == 'NoSuchBucket':
                    print(f"   🔧 Fix: Bucket '{self.bucket_name}' does not exist or is not accessible")
                elif error_code == 'AccessDenied':
                    print("   🔧 Fix: Insufficient permissions to access this bucket/prefix")
                else:
                    print(f"   🔧 Fix: Check your S3 configuration and credentials")
                
                continue
        

        return samples
    
    def _get_image_bytes_from_s3(self, s3_key: str) -> bytes:
        """Get raw image bytes from S3 (without decoding)"""
        # Check cache first (for preloaded bytes)
        if s3_key in self.bytes_cache:
            return self.bytes_cache[s3_key]
        
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            image_bytes = response['Body'].read()
            return image_bytes
        except ClientError as e:
            raise ValueError(f"Error loading image bytes '{s3_key}' from S3: {e}")
    
    def _decode_image_bytes(self, image_bytes: bytes) -> torch.Tensor:
        """Decode image bytes to tensor (happens in DataLoader worker)"""
        try:
            # Option 1: OpenCV → Direct Tensor (fastest)
            use_opencv_direct = True  # Set to True for maximum speed
            
            if use_opencv_direct:
                import cv2
                import numpy as np
                
                # Decode bytes to numpy array  
                nparr = np.frombuffer(image_bytes, np.uint8)
                # Decode image (returns BGR)
                image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise ValueError("cv2.imdecode failed")
                    
                # Convert BGR to RGB and resize to consistent dimensions
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image_rgb = cv2.resize(image_rgb, (960, 540), interpolation=cv2.INTER_LINEAR)
                
                # Convert to tensor and normalize (skip PIL transforms)
                tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
                # Apply ImageNet normalization directly
                return tensor
            else:
                # Option 2: OpenCV → PIL (good balance)
                import cv2
                import numpy as np
                
                nparr = np.frombuffer(image_bytes, np.uint8)
                image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise ValueError("cv2.imdecode failed")
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image_rgb)
                image = image.resize((960, 540), Image.LANCZOS)  # Ensure consistent size
                
                if self.transform:
                    image = self.transform(image)
                return image
                
        except Exception as e:
            # Return black fallback image if decoding fails
            print(f"Error decoding image bytes: {e}")
            if self.transform:
                fallback = self.transform(Image.new('RGB', (960, 540), color='black'))
            else:
                fallback = T.ToTensor()(Image.new('RGB', (960, 540), color='black'))
            return fallback
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _get_sample_bytes(self, idx: int) -> Tuple[List[bytes], List[str]]:
        """Get raw image bytes for a sample (happens in main process with S3 credentials)"""
        sequence_prefix, start_idx, image_keys = self.samples[idx]
        
        # Get consecutive image keys
        consecutive_keys = image_keys[start_idx:start_idx + self.total_frames]
        
        # Download raw bytes from S3
        image_bytes_list = []
        for key in consecutive_keys:
            try:
                image_bytes = self._get_image_bytes_from_s3(key)
                image_bytes_list.append(image_bytes)
            except Exception as e:
                print(f"Error downloading image bytes {key}: {e}")
                # Use empty bytes as placeholder for failed downloads
                image_bytes_list.append(b'')
        
        return image_bytes_list, consecutive_keys
    
    def _decode_sample_from_bytes(self, image_bytes_list: List[bytes], consecutive_keys: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode images from bytes (happens in DataLoader worker process)"""
        images = []
        
        for i, image_bytes in enumerate(image_bytes_list):
            if image_bytes:  # Non-empty bytes
                try:
                    decoded_image = self._decode_image_bytes(image_bytes)
                    images.append(decoded_image)
                except Exception as e:
                    print(f"Error decoding image bytes for key {consecutive_keys[i]}: {e}")
                    # Use fallback with consistent size
                    fallback = T.ToTensor()(Image.new('RGB', (960, 540), color='black'))
                    images.append(fallback)
            else:
                # Use fallback for empty bytes (failed downloads)
                fallback = T.ToTensor()(Image.new('RGB', (960, 540), color='black'))
                images.append(fallback)
        
        # Stack images and split into current/future
        images = torch.stack(images)  # [total_frames, C, H, W]
        
        current_frames = images[:self.m]  # [m, C, H, W]
        future_frames = images[self.m:]   # [n, C, H, W]
        
        return current_frames, future_frames
    
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample of consecutive images.
        
        For multiprocessing compatibility, this method downloads and decodes in one step.
        The bytes-based methods above can be used for custom caching strategies.
        
        Returns:
            current_frames: [m, C, H, W] tensor of input frames
            future_frames: [n, C, H, W] tensor of target frames
        """
        # Simple approach: download and decode immediately
        # For advanced users: use _get_sample_bytes + _decode_sample_from_bytes with custom caching
        image_bytes_list, consecutive_keys = self._get_sample_bytes(idx)
        return self._decode_sample_from_bytes(image_bytes_list, consecutive_keys)
    
    
    def get_sequence_info(self, idx: int) -> dict:
        """Get metadata about a specific sample"""
        sequence_prefix, start_idx, image_keys = self.samples[idx]
        consecutive_keys = image_keys[start_idx:start_idx + self.total_frames]
        
        return {
            'sequence_prefix': sequence_prefix,
            'start_index': start_idx,
            'total_frames': self.total_frames,
            'current_frame_keys': consecutive_keys[:self.m],
            'future_frame_keys': consecutive_keys[self.m:],
            'bucket_name': self.bucket_name
        }
    
    def visualize_sample(self, idx: int, save_path: Optional[str] = None, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Visualize a sample showing current and future frames.
        
        Args:
            idx: Sample index to visualize
            save_path: Optional path to save the visualization
            figsize: Figure size for the plot
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("matplotlib is required for visualization. Install with: pip install matplotlib")
            return
        
        # Get the sample
        current_frames, future_frames = self[idx]
        info = self.get_sequence_info(idx)
        
        # Denormalize images for display (reverse ImageNet normalization)
        def denormalize(tensor):
            # identity for now
            return tensor
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, max(self.m, self.n), figure=fig)
        
        # Add section headers
        fig.text(0.05, 0.75, '🟢 CURRENT FRAMES (Input)', fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        fig.text(0.05, 0.35, '🔵 FUTURE FRAMES (Target)', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        # Plot current frames
        for i in range(self.m):
            ax = fig.add_subplot(gs[0, i])
            
            # Denormalize and convert to displayable format
            img_tensor = denormalize(current_frames[i]).clamp(0, 1)
            img_array = img_tensor.permute(1, 2, 0).numpy()
            
            ax.imshow(img_array)
            ax.set_title(f'Input #{i+1}\n{os.path.basename(info["current_frame_keys"][i])}', fontsize=10)
            ax.axis('off')
            # Add green border for current frames
            for spine in ax.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(2)
                spine.set_visible(True)
        
        # Plot future frames
        for i in range(self.n):
            ax = fig.add_subplot(gs[1, i])
            
            # Denormalize and convert to displayable format
            img_tensor = denormalize(future_frames[i]).clamp(0, 1)
            img_array = img_tensor.permute(1, 2, 0).numpy()
            
            ax.imshow(img_array)
            ax.set_title(f'Target #{i+1}\n{os.path.basename(info["future_frame_keys"][i])}', fontsize=10)
            ax.axis('off')
            # Add blue border for future frames
            for spine in ax.spines.values():
                spine.set_edgecolor('blue')
                spine.set_linewidth(2)
                spine.set_visible(True)
        
        # Add main title
        plt.suptitle(f'S3Dataset Sample {idx} - Sequence: {info["sequence_prefix"]}\n'
                     f'Frames {info["start_index"]} to {info["start_index"] + self.total_frames - 1}', 
                     fontsize=14, y=0.95)
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📸 Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_batch(self, dataloader, num_samples: int = 4, save_path: Optional[str] = None) -> None:
        """
        Visualize multiple samples from a DataLoader batch.
        
        Args:
            dataloader: PyTorch DataLoader
            num_samples: Number of samples to visualize
            save_path: Optional path to save the visualization
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("matplotlib is required for visualization. Install with: pip install matplotlib")
            return
        
        # Get a batch
        batch_current, batch_future = next(iter(dataloader))
        batch_size = min(batch_current.size(0), num_samples)
        
        # Create figure
        fig = plt.figure(figsize=(20, batch_size * 4))
        
        # Denormalize function
        def denormalize(tensor):
            return tensor
        
        denorm_current = denormalize(batch_current[:batch_size]).clamp(0, 1)
        denorm_future = denormalize(batch_future[:batch_size]).clamp(0, 1)
        
        # Plot each sample in the batch
        for sample_idx in range(batch_size):
            # Current frames row
            for frame_idx in range(self.m):
                ax = plt.subplot(batch_size * 2, max(self.m, self.n), 
                               sample_idx * 2 * max(self.m, self.n) + frame_idx + 1)
                
                img_array = denorm_current[sample_idx, frame_idx].permute(1, 2, 0).numpy()
                ax.imshow(img_array)
                ax.set_title(f'Sample {sample_idx} - Current {frame_idx+1}', fontsize=8)
                ax.axis('off')
            
            # Future frames row
            for frame_idx in range(self.n):
                ax = plt.subplot(batch_size * 2, max(self.m, self.n),
                               sample_idx * 2 * max(self.m, self.n) + max(self.m, self.n) + frame_idx + 1)
                
                img_array = denorm_future[sample_idx, frame_idx].permute(1, 2, 0).numpy()
                ax.imshow(img_array)
                ax.set_title(f'Sample {sample_idx} - Future {frame_idx+1}', fontsize=8)
                ax.axis('off')
        
        plt.suptitle(f'DataLoader Batch Visualization ({batch_size} samples)', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📸 Batch visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_random_samples(self, num_samples: int = 6, save_path: Optional[str] = None, figsize: Tuple[int, int] = (20, 12)) -> None:
        """
        Visualize random samples from the dataset.
        
        Args:
            num_samples: Number of random samples to visualize
            save_path: Optional path to save the visualization
            figsize: Figure size for the plot
        """
        try:
            import matplotlib.pyplot as plt
            import random
        except ImportError:
            print("matplotlib is required for visualization. Install with: pip install matplotlib")
            return
        
        if len(self.samples) == 0:
            print("No samples available in dataset")
            return
        
        # Get random sample indices
        random_indices = random.sample(range(len(self.samples)), min(num_samples, len(self.samples)))
        actual_samples = len(random_indices)
        
        print(f"🎲 Visualizing {actual_samples} random samples from dataset (indices: {random_indices})")
        
        # Create figure - each sample gets 2 rows (current + future frames)
        fig = plt.figure(figsize=figsize)
        
        # Calculate grid: samples as columns, each with 2 rows (current/future)
        cols = actual_samples
        total_rows = max(self.m, self.n) * 2  # 2 sections: current and future
        
        # Denormalize function
        def denormalize(tensor):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            return tensor
        
        # Process each random sample
        for col_idx, sample_idx in enumerate(random_indices):
            try:
                # Get sample data
                current_frames, future_frames = self[sample_idx]
                info = self.get_sequence_info(sample_idx)
                
                # Plot current frames (top section)
                for frame_idx in range(self.m):
                    ax = plt.subplot(total_rows, cols, 
                                   frame_idx * cols + col_idx + 1)
                    
                    img_tensor = denormalize(current_frames[frame_idx]).clamp(0, 1)
                    img_array = img_tensor.permute(1, 2, 0).numpy()
                    
                    ax.imshow(img_array)
                    if frame_idx == 0:  # Add sample info on first frame
                        ax.set_title(f'Sample {sample_idx}\n🟢 Input {frame_idx+1}', fontsize=8)
                    else:
                        ax.set_title(f'🟢 Input {frame_idx+1}', fontsize=8)
                    ax.axis('off')
                    # Add green border for current frames
                    for spine in ax.spines.values():
                        spine.set_edgecolor('green')
                        spine.set_linewidth(1)
                        spine.set_visible(True)
                
                # Plot future frames (bottom section)  
                for frame_idx in range(self.n):
                    ax = plt.subplot(total_rows, cols,
                                   (self.m + frame_idx) * cols + col_idx + 1)
                    
                    img_tensor = denormalize(future_frames[frame_idx]).clamp(0, 1)
                    img_array = img_tensor.permute(1, 2, 0).numpy()
                    
                    ax.imshow(img_array)
                    ax.set_title(f'🔵 Target {frame_idx+1}', fontsize=8)
                    ax.axis('off')
                    # Add blue border for future frames
                    for spine in ax.spines.values():
                        spine.set_edgecolor('blue')
                        spine.set_linewidth(1)
                        spine.set_visible(True)
                    
            except Exception as e:
                print(f"Error loading sample {sample_idx}: {e}")
                continue
        
        plt.suptitle(f'Random Dataset Samples ({actual_samples} samples from {len(self.samples)} total)\n'
                     f'Dataset: {self.bucket_name} | Sequences: {len(self.sequence_prefixes)} | '
                     f'Frames per sample: {self.m} current + {self.n} future', 
                     fontsize=14, y=0.98)
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📸 Random samples visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()


def create_s3_dataset(
    bucket_name: str,
    sequence_prefixes: List[str],
    m: int = 3,
    n: int = 3,
    **kwargs
) -> S3Dataset:
    """
    Convenience function to create S3Dataset
    
    Args:
        bucket_name: S3 bucket name
        sequence_prefixes: List of sequence prefixes
        m: Number of input frames
        n: Number of target frames
        **kwargs: Additional arguments for dataset
    """
    return S3Dataset(
        bucket_name=bucket_name,
        sequence_prefixes=sequence_prefixes,
        m=m,
        n=n,
        **kwargs
    )

# Legacy alias for backward compatibility
S3ConsecutiveImagesDataset = S3Dataset
PrefetchingS3Dataset = S3Dataset  # Another legacy alias


# Example usage
if __name__ == "__main__":
    # Example of how to use the dataset
    
    # Define your S3 bucket and sequences
    bucket_name = "research-datasets"
    sequence_prefixes = [
        "autonomy_youtube/sf_day/",
        "autonomy_youtube/smoky_mountains/",
    ]

    import pdb; pdb.set_trace()
    
    try:
        # Create S3 dataset with bytes preloading for multiprocessing
        dataset = S3Dataset(
            bucket_name=bucket_name,
            sequence_prefixes=sequence_prefixes,
            m=3,  # 3 input frames
            n=3,  # 3 target frames
            image_extension=".png",
            preload_bytes=False  # Enable for num_workers > 0
        )
        print(f"Dataset created successfully with {len(dataset)} samples")
        
        # Test loading a sample
        if len(dataset) > 0:
            current_frames, future_frames = dataset[0]
            print(f"Sample shapes - Current: {current_frames.shape}, Future: {future_frames.shape}")
            
            # Get sample info
            info = dataset.get_sequence_info(0)
            print(f"Sample info: {info}")
            
            # Visualize images
            print("\nVisualizing sample images...")
            dataset.visualize_sample(0, save_path="sample_visualization.png")
            
            # Create DataLoader with prefetch_factor for efficient loading
            from torch.utils.data import DataLoader
            
            dataloader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=True,
                num_workers=4,          # Use multiple workers for parallel loading
                prefetch_factor=4,      # Prefetch 4 batches per worker (PyTorch 1.8+)
                pin_memory=True,         # Use pinned memory for faster GPU transfer,
            )
            
            print("\nTesting DataLoader...")
            for batch_idx, (X, y) in enumerate(dataloader):
                print(f"Batch {batch_idx}: X.shape={X.shape}, y.shape={y.shape}")
                if batch_idx >= 2:  # Test first few batches
                    break
            
            # Visualize a batch
            print("\nVisualizing DataLoader batch...")
            dataset.visualize_batch(dataloader, num_samples=2, save_path="batch_visualization.png")
            
            # Visualize random samples
            print("\nVisualizing random samples from dataset...")
            dataset.visualize_random_samples(num_samples=4, save_path="random_samples.png")
                    
    except Exception as e:
        print(f"Error: {e}")
        print("\nTo use this dataset:")
        print("1. Update bucket_name and sequence_prefixes")
        print("2. Ensure AWS credentials are configured")
        print("3. Verify S3 bucket structure matches expected format")