import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Tuple, Optional, Callable, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from botocore.config import Config
import io
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import logging
import threading
import queue
import time
import weakref
from collections import OrderedDict
import hashlib
import psutil
import gc

# Set up logging
logger = logging.getLogger(__name__)

class LRUCache:
    """Thread-safe LRU cache with memory-aware eviction."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 1024):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.memory_usage = 0
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
    
    def put(self, key: str, value: Any, size_bytes: int = 0) -> None:
        with self.lock:
            # Remove if already exists
            if key in self.cache:
                old_value = self.cache.pop(key)
                self.memory_usage -= getattr(old_value, '_cache_size', 0)
            
            # Add size info to value
            if hasattr(value, '__sizeof__'):
                size_bytes = max(size_bytes, value.__sizeof__())
            setattr(value, '_cache_size', size_bytes)
            
            # Add new value
            self.cache[key] = value
            self.memory_usage += size_bytes
            
            # Evict if necessary
            self._evict_if_needed()
    
    def _evict_if_needed(self) -> None:
        """Evict least recently used items if over limits."""
        while (len(self.cache) > self.max_size or 
               self.memory_usage > self.max_memory_bytes) and self.cache:
            
            # Remove least recently used (first item)
            key, value = self.cache.popitem(last=False)
            self.memory_usage -= getattr(value, '_cache_size', 0)
            
    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.memory_usage = 0
    
    def stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'memory_usage_mb': self.memory_usage / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024)
            }


class AsyncPrefetcher:
    """Asynchronous prefetcher for S3 images with intelligent batching."""
    
    def __init__(self, s3_client, bucket: str, max_workers: int = 32, 
                 prefetch_size: int = 100, batch_size: int = 10):
        self.s3_client = s3_client
        self.bucket = bucket
        self.max_workers = max_workers
        self.prefetch_size = prefetch_size
        self.batch_size = batch_size
        
        self.prefetch_queue = queue.Queue(maxsize=prefetch_size)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures: Dict[str, Future] = {}
        self.running = True
        
        # Statistics
        self.stats = {
            'requests': 0,
            'cache_hits': 0,
            'downloads': 0,
            'errors': 0,
            'total_bytes': 0
        }
        
    def prefetch_batch(self, s3_keys: List[str]) -> None:
        """Prefetch a batch of S3 keys."""
        if not self.running:
            return
            
        # Group keys into batches for efficient downloading
        for i in range(0, len(s3_keys), self.batch_size):
            batch_keys = s3_keys[i:i + self.batch_size]
            
            # Submit batch download
            future = self.executor.submit(self._download_batch, batch_keys)
            
            # Store futures for each key in the batch
            for key in batch_keys:
                self.futures[key] = future
    
    def _download_batch(self, s3_keys: List[str]) -> Dict[str, bytes]:
        """Download a batch of S3 objects efficiently."""
        results = {}
        
        # Create thread-local S3 client
        thread_s3 = boto3.client(
            's3',
            config=Config(
                retries={'max_attempts': 3},
                max_pool_connections=50
            )
        )
        
        for key in s3_keys:
            try:
                response = thread_s3.get_object(Bucket=self.bucket, Key=key)
                data = response['Body'].read()
                results[key] = data
                
                self.stats['downloads'] += 1
                self.stats['total_bytes'] += len(data)
                
            except Exception as e:
                logger.warning(f"Failed to download {key}: {e}")
                self.stats['errors'] += 1
                results[key] = None
                
        return results
    
    def get(self, s3_key: str, timeout: float = 10.0) -> Optional[bytes]:
        """Get data for S3 key, waiting for prefetch if necessary."""
        self.stats['requests'] += 1
        
        if s3_key in self.futures:
            try:
                # Get batch results
                batch_results = self.futures[s3_key].result(timeout=timeout)
                
                # Clean up future
                del self.futures[s3_key]
                
                return batch_results.get(s3_key)
                
            except Exception as e:
                logger.warning(f"Prefetch failed for {s3_key}: {e}")
                self.stats['errors'] += 1
                
        return None
    
    def shutdown(self) -> None:
        """Shutdown the prefetcher."""
        self.running = False
        self.executor.shutdown(wait=False)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get prefetcher statistics."""
        stats = self.stats.copy()
        stats['cache_hit_rate'] = (
            stats['cache_hits'] / max(stats['requests'], 1)
        )
        stats['queue_size'] = len(self.futures)
        return stats


class MemoryMonitor:
    """Monitor memory usage and trigger cleanup when needed."""
    
    def __init__(self, max_memory_percent: float = 80.0):
        self.max_memory_percent = max_memory_percent
        self.process = psutil.Process()
        
    def should_cleanup(self) -> bool:
        """Check if memory cleanup is needed."""
        try:
            memory_percent = self.process.memory_percent()
            return memory_percent > self.max_memory_percent
        except:
            return False
    
    def cleanup(self) -> None:
        """Force garbage collection and cleanup."""
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None



class S3ConsecutiveImagesDataset(Dataset):
    """
    High-performance PyTorch Dataset for streaming consecutive images from AWS S3.
    
    Features:
    - Intelligent prefetching with async downloads
    - Memory-aware LRU caching
    - Adaptive batch downloading
    - Connection pooling and retry logic
    - Memory monitoring and cleanup
    - Comprehensive performance metrics
    
    Args:
        s3_bucket (str): Name of the S3 bucket
        s3_prefix (str): S3 prefix/folder path (e.g., 'video_frames/amsterdam/')
        batch_size (int): Number of consecutive images to return in each batch
        transform (callable, optional): Optional transform to be applied to each image
        image_extensions (tuple): Valid image file extensions
        sort_key (callable, optional): Function to sort image filenames (default: natural sort)
        start_frame_idx (int): Starting frame index (default: 0)
        aws_access_key_id (str, optional): AWS access key ID (if not using IAM/env vars)
        aws_secret_access_key (str, optional): AWS secret access key (if not using IAM/env vars)
        aws_region (str): AWS region name (default: 'us-east-1')
        
        # Performance & Streaming Options
        streaming_mode (bool): Enable high-performance streaming mode (default: True)
        cache_size (int): Maximum number of images to cache in memory (default: 1000)
        cache_memory_mb (int): Maximum memory usage for cache in MB (default: 2048)
        max_workers (int): Number of threads for parallel S3 downloads (default: 64)
        prefetch_size (int): Number of batches to prefetch ahead (default: 10)
        prefetch_workers (int): Number of workers for prefetching (default: 32)
        adaptive_prefetch (bool): Dynamically adjust prefetch based on access patterns (default: True)
        memory_limit_percent (float): Trigger cleanup when memory usage exceeds this % (default: 80.0)
        connection_pool_size (int): Size of S3 connection pool (default: 100)
        enable_memory_monitor (bool): Enable automatic memory monitoring (default: True)
        
        # Debugging & Monitoring
        enable_stats (bool): Enable performance statistics collection (default: True)
        log_interval (int): Log performance stats every N batches (default: 100)
    """
    
    def __init__(
        self,
        s3_bucket: str,
        s3_prefix: str,
        batch_size: int,
        transform: Optional[Callable] = None,
        image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'),
        sort_key: Optional[Callable] = None,
        start_frame_idx: int = 300,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: str = 'us-phoenix-1',
        
        # Performance & Streaming Options
        streaming_mode: bool = True,
        cache_size: int = 1000,
        cache_memory_mb: int = 2048,
        max_workers: int = 64,
        prefetch_size: int = 10,
        prefetch_workers: int = 32,
        adaptive_prefetch: bool = True,
        memory_limit_percent: float = 80.0,
        connection_pool_size: int = 100,
        enable_memory_monitor: bool = True,
        
        # Debugging & Monitoring
        enable_stats: bool = True,
        log_interval: int = 100
    ):
        import pdb; pdb.set_trace()
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix.rstrip('/')  # Remove trailing slash
        self.batch_size = batch_size
        self.start_frame_idx = start_frame_idx

        if aws_access_key_id is None:
            aws_access_key_id = os.getenv("ACCESS_KEY_ID")

        if aws_secret_access_key is None:
            aws_secret_access_key = os.getenv("SECRET_ACCESS_KEY")
        
        # Store configuration
        self.streaming_mode = streaming_mode
        self.max_workers = max_workers
        self.prefetch_size = prefetch_size
        self.adaptive_prefetch = adaptive_prefetch
        self.enable_stats = enable_stats
        self.log_interval = log_interval
        
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
        ])
        self.image_extensions = image_extensions
        self.aws_region = aws_region
        
        # Initialize performance components
        if streaming_mode:
            self.cache = LRUCache(max_size=cache_size, max_memory_mb=cache_memory_mb)
            self.memory_monitor = MemoryMonitor(memory_limit_percent) if enable_memory_monitor else None
            self.prefetcher = None  # Will be initialized on first access
        else:
            self.cache = None
            self.memory_monitor = None
            self.prefetcher = None
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'prefetch_hits': 0,
            'download_time': 0.0,
            'transform_time': 0.0,
            'memory_cleanups': 0,
            'batch_times': []
        } if enable_stats else None
        
        self.access_pattern = []  # Track access patterns for adaptive prefetching
        self.last_accessed_idx = -1
        
        # Initialize S3 client with optimized configuration
        self.s3_client = self._init_s3_client(
            aws_access_key_id, aws_secret_access_key, aws_region, connection_pool_size
        )
        
        # Get all image files from S3
        logger.info(f"🔍 Scanning S3 bucket {s3_bucket} with prefix {s3_prefix}...")
        self.image_keys = self._get_s3_image_keys()
        
        # Sort files (default: natural sort for numbered sequences)
        if sort_key is None:
            self.image_keys = self._natural_sort(self.image_keys)
        else:
            self.image_keys.sort(key=sort_key)
        
        # If no images, make dataset empty and return early
        if len(self.image_keys) == 0:
            self._empty = True
            logger.warning(f"No images found in S3 bucket {s3_bucket} with prefix {s3_prefix}")
            return
        else:
            self._empty = False
            logger.info(f"✅ Found {len(self.image_keys)} images")
            
        # Validate start_frame_idx
        if start_frame_idx < 0:
            raise ValueError(f"start_frame_idx must be non-negative, got {start_frame_idx}")
        if start_frame_idx >= len(self.image_keys):
            raise ValueError(f"start_frame_idx ({start_frame_idx}) must be less than total number of images ({len(self.image_keys)})")
        
        # Check if we have enough images from start_frame_idx
        available_frames = len(self.image_keys) - start_frame_idx
        if available_frames < batch_size:
            raise ValueError(f"Not enough images from start_frame_idx {start_frame_idx}. Found {available_frames}, need at least {batch_size}")
        
        # Initialize prefetcher for streaming mode
        if streaming_mode:
            self.prefetcher = AsyncPrefetcher(
                self.s3_client, 
                self.s3_bucket, 
                max_workers=prefetch_workers,
                prefetch_size=prefetch_size * batch_size,
                batch_size=batch_size
            )
            logger.info(f"🚀 Streaming mode enabled with {prefetch_workers} prefetch workers")

    def _init_s3_client(self, access_key_id: Optional[str], secret_access_key: Optional[str], 
                       region: str, connection_pool_size: int = 100):
        """Initialize S3 client with optimized configuration."""
        try:
            # Optimized S3 client configuration
            config = Config(
                retries={'max_attempts': 3, 'mode': 'adaptive'},
                max_pool_connections=connection_pool_size,
                region_name=region,
                s3={
                    'max_concurrent_requests': connection_pool_size,
                    'max_bandwidth': None,  # No bandwidth limit
                    'use_accelerate_endpoint': False,  # Can be enabled for global access
                    'addressing_style': 'auto'
                }
            )
            
            if access_key_id and secret_access_key:
                # Use provided credentials
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id=access_key_id,
                    aws_secret_access_key=secret_access_key,
                    config=config
                )
            else:
                # Use default credentials (IAM role, environment vars, ~/.aws/credentials)
                s3_client = boto3.client('s3', config=config)
                
            # Test connection
            s3_client.head_bucket(Bucket=self.s3_bucket)
            logger.info(f"✅ Connected to S3 bucket: {self.s3_bucket} with {connection_pool_size} connections")
            return s3_client
            
        except NoCredentialsError:
            raise RuntimeError("AWS credentials not found. Please configure credentials.")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise RuntimeError(f"S3 bucket '{self.s3_bucket}' does not exist or you don't have access.")
            else:
                raise RuntimeError(f"Error accessing S3 bucket: {e}")

    def _get_s3_image_keys(self) -> List[str]:
        """Get all valid image file keys from S3 bucket with the given prefix."""
        try:
            image_keys = []
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            # List all objects with the given prefix
            for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=self.s3_prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        # Extract filename from key
                        filename = os.path.basename(key)
                        if filename.lower().endswith(self.image_extensions):
                            image_keys.append(key)
            
            if not image_keys:
                logger.warning(f"No valid image files found in S3 bucket '{self.s3_bucket}' with prefix '{self.s3_prefix}'")
                return []
            
            logger.info(f"Found {len(image_keys)} images in S3 bucket '{self.s3_bucket}' with prefix '{self.s3_prefix}'")
            return image_keys
            
        except ClientError as e:
            raise RuntimeError(f"Error listing S3 objects: {e}")

    def _natural_sort(self, key_list: List[str]) -> List[str]:
        """
        Sort S3 keys naturally based on filename.
        """
        import re
        
        def natural_key(s3_key):
            filename = os.path.basename(s3_key)
            return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', filename)]
        
        return sorted(key_list, key=natural_key)

    def _download_image_from_s3(self, s3_key: str) -> Image.Image:
        """Download a single image from S3 and return as PIL Image."""
        if self.cache_images and s3_key in self.image_cache:
            return self.image_cache[s3_key].copy()
        
        try:
            # Download image data from S3
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            image_data = response['Body'].read()
            
            # Load image from bytes
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Cache if enabled
            if self.cache_images:
                self.image_cache[s3_key] = image.copy()
            
            return image
            
        except ClientError as e:
            raise RuntimeError(f"Error downloading image from S3 key '{s3_key}': {e}")
        except Exception as e:
            raise RuntimeError(f"Error processing image from S3 key '{s3_key}': {e}")

    def _download_images_batch(self, s3_keys: List[str]) -> List[Image.Image]:
        """Download multiple images from S3 in parallel."""
        images = [None] * len(s3_keys)
        
        def download_single(idx_key):
            idx, key = idx_key
            return idx, self._download_image_from_s3(key)
        
        # Download images in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {executor.submit(download_single, (i, key)): i 
                           for i, key in enumerate(s3_keys)}
            
            for future in as_completed(future_to_idx):
                try:
                    idx, image = future.result()
                    images[idx] = image
                except Exception as e:
                    idx = future_to_idx[future]
                    raise RuntimeError(f"Failed to download image at index {idx}: {e}")
        
        return images

    def __len__(self) -> int:
        if hasattr(self, '_empty') and self._empty:
            return 0
        return len(self.image_keys) - self.start_frame_idx - self.batch_size + 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        if hasattr(self, '_empty') and self._empty:
            raise IndexError("Empty dataset: no images available.")
        
        """
        High-performance streaming get item with intelligent caching and prefetching.
        
        Args:
            idx (int): Starting index for the batch
            
        Returns:
            torch.Tensor: Tensor of shape (batch_size, C, H, W) containing consecutive images
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")
        
        batch_start_time = time.time()
        
        # Update statistics
        if self.stats:
            self.stats['total_requests'] += 1
        
        # Memory cleanup if needed
        if self.memory_monitor and self.memory_monitor.should_cleanup():
            self._perform_memory_cleanup()
        
        # Get consecutive S3 keys starting from start_frame_idx
        actual_idx = self.start_frame_idx + idx
        batch_keys = self.image_keys[actual_idx:actual_idx + self.batch_size]
        
        # Adaptive prefetching based on access patterns
        if self.streaming_mode and self.adaptive_prefetch:
            self._adaptive_prefetch(idx, actual_idx)
        
        # Download images with streaming optimizations
        if self.streaming_mode:
            images = self._download_images_streaming(batch_keys)
        else:
            images = self._download_images_batch(batch_keys)
        
        # Apply transforms with timing
        transform_start = time.time()
        transformed_images = []
        for image in images:
            if self.transform:
                image = self.transform(image)
            transformed_images.append(image)
        
        if self.stats:
            self.stats['transform_time'] += time.time() - transform_start
        
        # Stack images into a batch tensor
        batch_tensor = torch.stack(transformed_images)  # Shape: (batch_size, C, H, W)
        
        # Update access pattern tracking
        self.last_accessed_idx = idx
        if len(self.access_pattern) > 100:  # Keep recent history
            self.access_pattern = self.access_pattern[-50:]
        self.access_pattern.append(idx)
        
        # Update batch timing statistics
        if self.stats:
            batch_time = time.time() - batch_start_time
            self.stats['batch_times'].append(batch_time)
            if len(self.stats['batch_times']) > 1000:
                self.stats['batch_times'] = self.stats['batch_times'][-500:]
            
            # Log performance stats periodically
            if self.stats['total_requests'] % self.log_interval == 0:
                self._log_performance_stats()
        
        return batch_tensor
    
    def _download_images_streaming(self, s3_keys: List[str]) -> List[Image.Image]:
        """Download images with streaming optimizations."""
        download_start = time.time()
        images = []
        cache_hits = 0
        prefetch_hits = 0
        
        for s3_key in s3_keys:
            image = None
            
            # Try cache first
            if self.cache:
                cached_data = self.cache.get(s3_key)
                if cached_data is not None:
                    try:
                        image = Image.open(io.BytesIO(cached_data)).convert('RGB')
                        cache_hits += 1
                    except Exception as e:
                        logger.warning(f"Failed to load cached image {s3_key}: {e}")
                        self.cache.put(s3_key, None)  # Remove invalid cache entry
            
            # Try prefetcher if cache miss
            if image is None and self.prefetcher:
                prefetch_data = self.prefetcher.get(s3_key, timeout=5.0)
                if prefetch_data is not None:
                    try:
                        image = Image.open(io.BytesIO(prefetch_data)).convert('RGB')
                        prefetch_hits += 1
                        
                        # Cache the data for future use
                        if self.cache:
                            self.cache.put(s3_key, prefetch_data, len(prefetch_data))
                            
                    except Exception as e:
                        logger.warning(f"Failed to load prefetched image {s3_key}: {e}")
            
            # Fallback to direct download
            if image is None:
                try:
                    image_data = self._download_single_image(s3_key)
                    image = Image.open(io.BytesIO(image_data)).convert('RGB')
                    
                    # Cache the data
                    if self.cache:
                        self.cache.put(s3_key, image_data, len(image_data))
                        
                except Exception as e:
                    logger.error(f"Failed to download image {s3_key}: {e}")
                    # Create a black placeholder image
                    image = Image.new('RGB', (224, 224), color='black')
            
            images.append(image)
        
        # Update statistics
        if self.stats:
            self.stats['download_time'] += time.time() - download_start
            self.stats['cache_hits'] += cache_hits
            self.stats['cache_misses'] += len(s3_keys) - cache_hits
            self.stats['prefetch_hits'] += prefetch_hits
        
        return images
    
    def _download_single_image(self, s3_key: str) -> bytes:
        """Download a single image with retry logic."""
        for attempt in range(3):
            try:
                response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
                return response['Body'].read()
            except Exception as e:
                if attempt == 2:
                    raise e
                time.sleep(0.1 * (2 ** attempt))
        
    def _adaptive_prefetch(self, current_idx: int, current_actual_idx: int) -> None:
        """Intelligently prefetch based on access patterns."""
        if not self.prefetcher or len(self.access_pattern) < 2:
            return
        
        # Analyze access pattern to predict next accesses
        recent_pattern = self.access_pattern[-10:]
        
        # Calculate typical step size
        if len(recent_pattern) >= 2:
            steps = [recent_pattern[i+1] - recent_pattern[i] for i in range(len(recent_pattern)-1)]
            avg_step = sum(steps) / len(steps)
            
            # Predict next few indices
            predicted_indices = []
            for i in range(1, self.prefetch_size + 1):
                next_idx = int(current_idx + avg_step * i)
                if next_idx < len(self):
                    predicted_indices.append(next_idx)
            
            # Convert to actual S3 keys and prefetch
            prefetch_keys = []
            for pred_idx in predicted_indices:
                actual_idx = self.start_frame_idx + pred_idx
                batch_keys = self.image_keys[actual_idx:actual_idx + self.batch_size]
                prefetch_keys.extend(batch_keys)
            
            if prefetch_keys:
                self.prefetcher.prefetch_batch(prefetch_keys)
    
    def _perform_memory_cleanup(self) -> None:
        """Perform memory cleanup when usage is high."""
        if self.cache:
            # Clear half of the cache
            cache_stats = self.cache.stats()
            if cache_stats['size'] > 0:
                self.cache.clear()
                logger.info(f"🧹 Memory cleanup: cleared cache ({cache_stats['memory_usage_mb']:.1f}MB freed)")
        
        # Force garbage collection
        if self.memory_monitor:
            self.memory_monitor.cleanup()
        
        if self.stats:
            self.stats['memory_cleanups'] += 1
    
    def _log_performance_stats(self) -> None:
        """Log performance statistics."""
        if not self.stats:
            return
        
        cache_hit_rate = self.stats['cache_hits'] / max(self.stats['cache_misses'] + self.stats['cache_hits'], 1)
        prefetch_hit_rate = self.stats['prefetch_hits'] / max(self.stats['total_requests'], 1)
        avg_batch_time = sum(self.stats['batch_times'][-100:]) / min(len(self.stats['batch_times']), 100)
        avg_download_time = self.stats['download_time'] / max(self.stats['total_requests'], 1)
        avg_transform_time = self.stats['transform_time'] / max(self.stats['total_requests'], 1)
        
        logger.info(f"📊 Performance Stats (last {self.log_interval} requests):")
        logger.info(f"   Cache hit rate: {cache_hit_rate:.2%}")
        logger.info(f"   Prefetch hit rate: {prefetch_hit_rate:.2%}")
        logger.info(f"   Avg batch time: {avg_batch_time:.3f}s")
        logger.info(f"   Avg download time: {avg_download_time:.3f}s")
        logger.info(f"   Avg transform time: {avg_transform_time:.3f}s")
        
        if self.cache:
            cache_stats = self.cache.stats()
            logger.info(f"   Cache: {cache_stats['size']}/{cache_stats['max_size']} items, {cache_stats['memory_usage_mb']:.1f}MB")
        
        if self.prefetcher:
            prefetch_stats = self.prefetcher.get_stats()
            logger.info(f"   Prefetcher: {prefetch_stats['queue_size']} pending, {prefetch_stats['total_bytes']/(1024*1024):.1f}MB downloaded")

    def get_image_info(self) -> dict:
        """Get comprehensive information about the dataset."""
        info = {
            'total_images': len(self.image_keys),
            'start_frame_idx': self.start_frame_idx,
            'available_images': len(self.image_keys) - self.start_frame_idx,
            'batch_size': self.batch_size,
            'num_batches': len(self),
            's3_bucket': self.s3_bucket,
            's3_prefix': self.s3_prefix,
            'streaming_mode': self.streaming_mode,
            'sample_keys': self.image_keys[self.start_frame_idx:self.start_frame_idx + 5]  # Show first 5 keys
        }
        
        if self.streaming_mode:
            info.update({
                'max_workers': self.max_workers,
                'prefetch_size': self.prefetch_size,
                'adaptive_prefetch': self.adaptive_prefetch,
            })
            
            if self.cache:
                info['cache_stats'] = self.cache.stats()
            
            if self.prefetcher:
                info['prefetcher_stats'] = self.prefetcher.get_stats()
        
        if self.stats:
            info['performance_stats'] = self.get_performance_stats()
        
        return info
    
    def get_performance_stats(self) -> dict:
        """Get detailed performance statistics."""
        if not self.stats:
            return {}
        
        total_requests = max(self.stats['total_requests'], 1)
        cache_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        
        return {
            'total_requests': self.stats['total_requests'],
            'cache_hit_rate': self.stats['cache_hits'] / max(cache_requests, 1),
            'prefetch_hit_rate': self.stats['prefetch_hits'] / total_requests,
            'avg_download_time': self.stats['download_time'] / total_requests,
            'avg_transform_time': self.stats['transform_time'] / total_requests,
            'avg_batch_time': sum(self.stats['batch_times'][-100:]) / min(len(self.stats['batch_times']), 100) if self.stats['batch_times'] else 0,
            'memory_cleanups': self.stats['memory_cleanups'],
            'total_batches_processed': len(self.stats['batch_times'])
        }
    
    def warmup_cache(self, num_batches: int = 10) -> None:
        """Warm up the cache by prefetching the first few batches."""
        if not self.streaming_mode or not self.prefetcher:
            logger.warning("Warmup only available in streaming mode")
            return
        
        logger.info(f"🔥 Warming up cache with {num_batches} batches...")
        
        warmup_keys = []
        for i in range(min(num_batches, len(self))):
            actual_idx = self.start_frame_idx + i
            batch_keys = self.image_keys[actual_idx:actual_idx + self.batch_size]
            warmup_keys.extend(batch_keys)
        
        if warmup_keys:
            self.prefetcher.prefetch_batch(warmup_keys)
            logger.info(f"✅ Cache warmup initiated for {len(warmup_keys)} images")
    
    def optimize_for_sequential_access(self) -> None:
        """Optimize dataset configuration for sequential access patterns."""
        if self.streaming_mode and self.prefetcher:
            # Increase prefetch size for sequential access
            self.prefetch_size = min(50, len(self) // 10)
            logger.info(f"🎯 Optimized for sequential access: prefetch_size={self.prefetch_size}")
    
    def optimize_for_random_access(self) -> None:
        """Optimize dataset configuration for random access patterns."""
        if self.streaming_mode:
            # Increase cache size, reduce prefetch for random access
            self.prefetch_size = max(5, self.prefetch_size // 2)
            if self.cache:
                # Increase cache size
                self.cache.max_size = min(2000, self.cache.max_size * 2)
            logger.info(f"🎲 Optimized for random access: prefetch_size={self.prefetch_size}")
    
    def clear_cache(self) -> None:
        """Clear the image cache to free memory."""
        if self.cache:
            cache_stats = self.cache.stats()
            self.cache.clear()
            logger.info(f"🧹 Cache cleared: {cache_stats['memory_usage_mb']:.1f}MB freed")
        
        if self.prefetcher:
            # Clear prefetcher queue
            self.prefetcher.futures.clear()
            logger.info("🧹 Prefetcher queue cleared")
    
    def __del__(self):
        """Cleanup resources when dataset is destroyed."""
        if hasattr(self, 'prefetcher') and self.prefetcher:
            self.prefetcher.shutdown()


# Utility functions
def create_s3_dataloader(
    s3_bucket: str,
    s3_prefix: str,
    batch_size: int,
    dataloader_batch_size: int = 1,
    transform: Optional[Callable] = None,
    shuffle: bool = False,
    num_workers: int = 0,
    start_frame_idx: int = 0,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_region: str = 'us-east-1',
    
    # Streaming Performance Options
    streaming_mode: bool = True,
    cache_size: int = 1000,
    cache_memory_mb: int = 2048,
    max_s3_workers: int = 64,
    prefetch_size: int = 10,
    prefetch_workers: int = 32,
    adaptive_prefetch: bool = True,
    memory_limit_percent: float = 80.0,
    connection_pool_size: int = 100,
    enable_memory_monitor: bool = True,
    enable_stats: bool = True,
    log_interval: int = 100,
    
    # Optimization presets
    access_pattern: str = 'sequential'  # 'sequential', 'random', or 'mixed'
) -> torch.utils.data.DataLoader:
    """
    Create a high-performance DataLoader for streaming S3 data.
    
    Args:
        s3_bucket: S3 bucket name
        s3_prefix: S3 prefix/folder path
        batch_size: Number of consecutive images per sample
        dataloader_batch_size: Number of samples per DataLoader batch
        transform: Optional image transform
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for loading
        start_frame_idx: Starting frame index
        aws_access_key_id: AWS access key ID (optional)
        aws_secret_access_key: AWS secret access key (optional)
        aws_region: AWS region name
        
        # Streaming Performance Options
        streaming_mode: Enable high-performance streaming
        cache_size: Maximum number of images to cache
        cache_memory_mb: Maximum memory for cache (MB)
        max_s3_workers: Number of threads for S3 downloads
        prefetch_size: Number of batches to prefetch
        prefetch_workers: Number of prefetch workers
        adaptive_prefetch: Enable intelligent prefetching
        memory_limit_percent: Memory usage threshold for cleanup
        connection_pool_size: S3 connection pool size
        enable_memory_monitor: Enable memory monitoring
        enable_stats: Enable performance statistics
        log_interval: Stats logging frequency
        
        access_pattern: Optimization preset ('sequential', 'random', 'mixed')
    
    Returns:
        torch.utils.data.DataLoader: Optimized DataLoader
    """
    
    # Apply access pattern optimizations
    if access_pattern == 'sequential':
        prefetch_size = max(prefetch_size, 20)
        adaptive_prefetch = True
        cache_size = min(cache_size, 500)  # Smaller cache for sequential
    elif access_pattern == 'random':
        prefetch_size = max(5, prefetch_size // 2)
        cache_size = max(cache_size, 2000)  # Larger cache for random
        cache_memory_mb = max(cache_memory_mb, 4096)
    elif access_pattern == 'mixed':
        # Balanced settings for mixed patterns
        prefetch_size = max(prefetch_size, 15)
        cache_size = max(cache_size, 1500)
    
    dataset = S3ConsecutiveImagesDataset(
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        batch_size=batch_size,
        transform=transform,
        start_frame_idx=start_frame_idx,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_region=aws_region,
        
        # Performance options
        streaming_mode=streaming_mode,
        cache_size=cache_size,
        cache_memory_mb=cache_memory_mb,
        max_workers=max_s3_workers,
        prefetch_size=prefetch_size,
        prefetch_workers=prefetch_workers,
        adaptive_prefetch=adaptive_prefetch,
        memory_limit_percent=memory_limit_percent,
        connection_pool_size=connection_pool_size,
        enable_memory_monitor=enable_memory_monitor,
        enable_stats=enable_stats,
        log_interval=log_interval
    )
    
    # Warm up cache for better initial performance
    if streaming_mode and access_pattern in ['sequential', 'mixed']:
        dataset.warmup_cache(num_batches=min(10, len(dataset) // 10))
    
    # Apply pattern-specific optimizations
    if access_pattern == 'sequential':
        dataset.optimize_for_sequential_access()
    elif access_pattern == 'random':
        dataset.optimize_for_random_access()
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=dataloader_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )


# Example transforms
def get_default_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Get default image transforms for training."""
    return transforms.Compose([
        # transforms.Resize(image_size),
        transforms.ToTensor(),
    ])


def get_video_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Get transforms optimized for video/temporal data."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Test S3ConsecutiveImagesDataset")
    parser.add_argument("--s3_bucket", type=str, required=True, help="S3 bucket name")
    parser.add_argument("--s3_prefix", type=str, required=True, help="S3 prefix/folder path")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of consecutive images per batch")
    parser.add_argument("--dataloader_batch_size", type=int, default=2, help="DataLoader batch size")
    parser.add_argument("--start_frame_idx", type=int, default=300, help="Starting frame index")
    parser.add_argument("--aws_region", type=str, default='us-east-1', help="AWS region")
    parser.add_argument("--cache_images", action='store_true', help="Cache images in memory")
    parser.add_argument("--max_s3_workers", type=int, default=4, help="Number of S3 download threads")
    
    args = parser.parse_args()
    import pdb; pdb.set_trace()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create dataset
    dataset = S3ConsecutiveImagesDataset(
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        batch_size=args.batch_size,
        transform=get_default_transforms(),
        start_frame_idx=args.start_frame_idx,
        aws_region=args.aws_region,
        cache_images=args.cache_images,
        max_workers=args.max_s3_workers,
        
    )

    # Print dataset info
    print("Dataset Info:")
    info = dataset.get_image_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Create dataloader
    dataloader = create_s3_dataloader(
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        batch_size=args.batch_size,
        dataloader_batch_size=args.dataloader_batch_size,
        shuffle=True,
        num_workers=2,
        start_frame_idx=args.start_frame_idx,
        aws_region=args.aws_region,
        cache_images=args.cache_images,
        max_s3_workers=args.max_s3_workers
    )
    
    # Test loading a few batches
    print(f"\nTesting DataLoader:")
    for i, batch in enumerate(dataloader):
        print(f"  Batch {i}: shape {batch.shape}")
        if i >= 2:  # Only show first 3 batches
            break
    
    print("\nDone!")