import os
import json
import pickle
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from botocore.config import Config
import io
from typing import List, Optional, Tuple, Dict
import re
from pathlib import Path
from collections import defaultdict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from tqdm import tqdm
import threading
from queue import Queue
import asyncio
import nest_asyncio
# Allow nested event loops (needed for Jupyter/DataLoader compatibility)
nest_asyncio.apply()

# from .youtube_shard_dataset import YouTubeVideoDataset
import torch.nn.functional as F


YOUTUBE_S3_CACHE = 'youtube_cache/youtube_dataset_df7b4701e6ade36698417531f6d163f2.pkl'


class YouTubeS3Dataset(Dataset):
    """
    S3 dataset for YouTube channel structure.
    
    Expected S3 structure:
    s3://bucket/openDV-YouTube/full_images/
    ├── channel1/
    │   ├── video1/
    │   │   ├── frame_00000.jpg
    │   │   ├── frame_00001.jpg
    │   │   └── ...
    │   ├── video2/
    │   │   └── ...
    │   └── ...
    ├── channel2/
    │   └── ...
    └── ...
    """
    
    def __init__(
        self,
        bucket_name: str = "research-datasets",
        root_prefix: str = "openDV-YouTube/full_images/",
        m: int = 3,
        n: int = 3,
        transform: Optional[T.Compose] = None,
        region_name: str = "us-east-1",
        cache_dir: str = "./youtube_dataset_cache",
        refresh_cache: bool = False,
        min_sequence_length: int = 50,  # Minimum frames in a video to be included
        skip_frames: int = 0,  # Skip first N frames of each video
        max_workers: int = 8,  # For parallel S3 operations
        verbose: bool = True,
        use_async: bool = False,  # Use async I/O for frame loading
        frame_sampling_rate: int = 1  # Sample every Nth frame (1=10Hz, 5=2Hz)
    ):
        """
        Args:
            bucket_name: S3 bucket name
            root_prefix: Root prefix for YouTube dataset
            m: Number of input frames
            n: Number of target frames  
            transform: Optional transforms to apply to images
            region_name: AWS region
            cache_dir: Directory to cache file listings
            refresh_cache: If True, refresh the cache even if it exists
            min_sequence_length: Minimum video length to include
            skip_frames: Number of frames to skip at the beginning of each video
            max_workers: Number of parallel workers for S3 operations
            verbose: Print detailed progress
            use_async: Use async I/O for frame loading (faster for many concurrent downloads)
            frame_sampling_rate: Sample every Nth frame (1=all frames/10Hz, 5=every 5th/2Hz)
        """
        self.bucket_name = bucket_name
        self.root_prefix = root_prefix.rstrip('/') + '/'
        self.m = m
        self.n = n
        self.total_frames = m + n
        self.transform = transform or self._default_transform()
        self.region_name = region_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Adjust minimum sequence length based on sampling rate
        frames_needed = (self.total_frames - 1) * frame_sampling_rate + 1 + skip_frames
        self.min_sequence_length = max(min_sequence_length, frames_needed)
        self.skip_frames = skip_frames
        self.max_workers = max_workers
        self.verbose = verbose
        self.use_async = use_async
        self.frame_sampling_rate = frame_sampling_rate
        
        # Initialize background caching system
        self.enable_background_cache = False  # Disabled to prevent memory growth
        self.ram_cache = {}  # {frame_key: image_bytes}
        self.cache_access_count = defaultdict(int)
        self.cache_lock = threading.Lock()
        self.max_cache_size_mb = 10240  # 10GB RAM cache (reduced from 100GB)
        self.current_cache_size = 0
        
        # Background caching
        self.prefetch_queue = Queue(maxsize=10000)  # Reduced from 1M to 10K
        self.cache_stop_event = threading.Event()
        self.background_cache_thread = None
        
        # Initialize S3 client with better configuration
        # Get endpoint URL from environment if available
        endpoint_url = os.environ.get('AWS_ENDPOINT_URL')
        
        client_kwargs = {
            'region_name': 'us-phoenix-1',  # Use Oracle Cloud region
            'config': Config(
                max_pool_connections=50,
                retries={
                    'max_attempts': 3,
                    'mode': 'adaptive'
                }
            )
        }
        
        # Add endpoint URL if provided (for S3-compatible services)
        if endpoint_url:
            client_kwargs['endpoint_url'] = endpoint_url
            if self.verbose:
                print(f"   Using S3-compatible endpoint: {endpoint_url}")
        
        # Use credentials from environment
        aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        
        if aws_access_key and aws_secret_key:
            client_kwargs.update({
                'aws_access_key_id': aws_access_key,
                'aws_secret_access_key': aws_secret_key,
            })
        
        self.s3_client = boto3.client('s3', **client_kwargs)
        
        # Initialize RAM cache system
        self.ram_cache = {}  # {frame_key: image_bytes}
        self.cache_access_count = defaultdict(int)
        self.cache_lock = threading.Lock()
        self.max_cache_size_mb = 10240  # 10GB RAM cache (reduced from 200GB)
        self.current_cache_size = 0
        
        # Background prefetching
        self.prefetch_queue = Queue(maxsize=10000)  # Reduced from 100K to 10K
        self.cache_stop_event = threading.Event()
        
        # Build or load the dataset index
        self.sequences = self._build_dataset_index(refresh_cache)
        
        # Start background caching thread only if enabled
        if self.enable_background_cache:
            self.background_cache_thread = threading.Thread(
                target=self._background_cache_worker, 
                daemon=True,
                name="BackgroundCacheWorker"
            )
            self.background_cache_thread.start()
        else:
            self.background_cache_thread = None
        
        if self.verbose:
            print(f"\n📊 YouTube S3 Dataset Statistics:")
            print(f"   Total channels: {len(self._get_channel_stats())}")
            print(f"   Total videos: {len(self._get_video_stats())}")
            print(f"   Total valid sequences: {len(self.sequences)}")
            print(f"   Total frames available: {sum(len(seq['frames']) for seq in self.sequences)}")
            cache_usage = (self.current_cache_size / (1024*1024)) if self.current_cache_size > 0 else 0
            print(f"   Background RAM cache: {cache_usage:.1f}MB / {self.max_cache_size_mb}MB")
            print(f"   Frame loading mode: {'Async I/O' if self.use_async else f'ThreadPool ({self.max_workers} workers)'}")
            print(f"   RAM cache: {self.max_cache_size_mb}MB")
            if self.frame_sampling_rate > 1:
                print(f"   Frame sampling: every {self.frame_sampling_rate} frames ({10.0/self.frame_sampling_rate:.1f} Hz)")
    
    def _default_transform(self):
        """Default transform if none provided"""
        return T.Compose([
            T.Resize((294, 518)),
            T.ToTensor(),
        ])
    
    def _get_cache_path(self) -> Path:
        """Generate cache file path based on dataset parameters"""
        # print parts that make up the cache key
        print(f"Generating cache key with:")
        print(f"   Bucket: {self.bucket_name}")
        print(f"   Root Prefix: {self.root_prefix}")
        print(f"   Min Sequence Length: {self.min_sequence_length}")
        print(f"   Skip Frames: {self.skip_frames}")
        cache_key = hashlib.md5(
            f"{self.bucket_name}_{self.root_prefix}_{self.min_sequence_length}_{self.skip_frames}".encode()
        ).hexdigest()

        return self.cache_dir / f"youtube_dataset_{cache_key}.pkl"
    
    def _build_dataset_index(self, refresh_cache: bool) -> List[Dict]:
        """Build or load the dataset index"""

        cache_path = self._get_cache_path()
        print(cache_path)
        
        # Also check if the hardcoded cache file exists
        hardcoded_cache_path = Path(YOUTUBE_S3_CACHE)
        if not refresh_cache and hardcoded_cache_path.exists():
            if self.verbose:
                print(f"📁 Found hardcoded cache file: {YOUTUBE_S3_CACHE}")
                print(f"📁 Loading dataset index from: {hardcoded_cache_path}")
            with open(hardcoded_cache_path, 'rb') as f:
                sequences = pickle.load(f)
            print(f"✅ Loaded {len(sequences)} sequences from hardcoded cache")
            return sequences
        
        # Try to load from cache
        if not refresh_cache and cache_path.exists():
            if self.verbose:
                print(f"📁 Loading dataset index from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                sequences = pickle.load(f)
            print(f"✅ Loaded {len(sequences)} sequences from cache")
            return sequences
        
        # Build index from S3
        if self.verbose:
            print(f"🔍 Building dataset index from S3...")
            print(f"   Bucket: {self.bucket_name}")
            print(f"   Root: {self.root_prefix}")
        
        sequences = []
        channel_videos = defaultdict(list)
        
        # First, discover all channels and videos
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        # Get all channels
        channels = set()
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.root_prefix, Delimiter='/'):
            if 'CommonPrefixes' in page:
                for prefix in page['CommonPrefixes']:
                    channel = prefix['Prefix'].replace(self.root_prefix, '').rstrip('/')
                    if channel:
                        channels.add(channel)
        
        if self.verbose:
            print(f"   Found {len(channels)} channels")
        
        print(channels)
        # Process each channel in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_channel = {
                executor.submit(self._process_channel, channel): channel 
                for channel in sorted(channels)
            }
            
            # Add progress bar for channel processing
            channel_progress = tqdm(as_completed(future_to_channel), 
                                  total=len(channels), 
                                  desc="Processing channels",
                                  disable=not self.verbose)
            
            for future in channel_progress:
                channel = future_to_channel[future]
                try:
                    channel_sequences = future.result()
                    sequences.extend(channel_sequences)
                    if self.verbose and channel_sequences:
                        channel_progress.set_postfix({"current": channel, "videos": len(channel_sequences)})
                except Exception as e:
                    if self.verbose:
                        channel_progress.set_postfix({"current": channel, "error": str(e)[:20]})
        
        # Save to cache
        with open(cache_path, 'wb') as f:
            pickle.dump(sequences, f)
        
        if self.verbose:
            print(f"💾 Saved dataset index to cache: {cache_path}")
        
        return sequences
    
    def _process_channel(self, channel: str) -> List[Dict]:
        """Process a single channel and return its valid video sequences"""
        channel_sequences = []
        channel_prefix = f"{self.root_prefix}{channel}/"
        
        # Get all videos in this channel
        videos = set()
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=channel_prefix, Delimiter='/'):
            if 'CommonPrefixes' in page:
                for prefix in page['CommonPrefixes']:
                    video = prefix['Prefix'].replace(channel_prefix, '').rstrip('/')
                    if video:
                        videos.add(video)
        
        # Process each video with progress bar
        video_progress = tqdm(sorted(videos), 
                             desc=f"Processing {channel}", 
                             leave=False,
                             disable=len(videos) < 5)  # Only show for channels with many videos
        
        for video in video_progress:
            video_prefix = f"{channel_prefix}{video}/"
            frames = self._get_video_frames(video_prefix)
            
            # Adjust minimum sequence length for frame sampling
            frames_needed = self.skip_frames + (self.total_frames - 1) * self.frame_sampling_rate + 1
            if len(frames) >= frames_needed:
                # Skip first N frames and create sequence entry
                valid_frames = frames[self.skip_frames:]
                if len(valid_frames) >= (self.total_frames - 1) * self.frame_sampling_rate + 1:
                    channel_sequences.append({
                        'channel': channel,
                        'video': video,
                        'frames': valid_frames,
                        'prefix': video_prefix
                    })
                    video_progress.set_postfix({"valid_videos": len(channel_sequences)})
        
        return channel_sequences
    
    def _get_video_frames(self, video_prefix: str) -> List[str]:
        """Get all frame paths for a video, sorted by frame number"""
        frames = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=video_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.lower().endswith(('.jpg', '.jpeg', '.png')):
                        frames.append(key)
        
        # Sort frames by extracting frame number
        def extract_frame_number(frame_path):
            filename = os.path.basename(frame_path)
            match = re.search(r'(\d+)', filename)
            return int(match.group(1)) if match else 0
        
        frames.sort(key=extract_frame_number)
        return frames
    
    def _background_downloader(self):
        """Background thread that downloads frames in batches"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while True:
                try:
                    # Get batch of frames to download
                    batch_request = self.download_queue.get(timeout=1)
                    if batch_request is None:  # Shutdown signal
                        break
                    
                    frame_keys, seq_info = batch_request
                    
                    # Download frames in parallel
                    future_to_key = {
                        executor.submit(self._download_frame, key): key 
                        for key in frame_keys
                    }
                    
                    for future in as_completed(future_to_key):
                        key = future_to_key[future]
                        try:
                            image_data = future.result()
                            if image_data:
                                with self.cache_lock:
                                    self.frame_cache[key] = image_data
                        except Exception as e:
                            print(f"Error downloading {key}: {e}")
                
                except:
                    continue
    
    def _download_frame(self, frame_key: str) -> Optional[bytes]:
        """Download a single frame from S3 and optionally cache locally"""
        # Check local disk cache first
        local_path = self.local_cache_dir / frame_key.replace('/', '_')
        if local_path.exists():
            try:
                with open(local_path, 'rb') as f:
                    return f.read()
            except:
                pass
        
        # Download from S3
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=frame_key)
            image_bytes = response['Body'].read()
            
            # Save to local cache
            try:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, 'wb') as f:
                    f.write(image_bytes)
            except:
                pass  # Ignore cache write errors
            
            return image_bytes
        except Exception as e:
            print(f"Error downloading frame {frame_key}: {e}")
            return None
    
    def _download_single_frame(self, frame_key: str) -> Optional[torch.Tensor]:
        """Download and process a single frame"""
        try:
            # Check RAM cache first
            image_bytes = self._get_frame_from_ram_cache(frame_key)
            
            if image_bytes is None:
                # Download from S3
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=frame_key)
                image_bytes = response['Body'].read()
                
                # Add to RAM cache for future use
                self._add_to_ram_cache(frame_key, image_bytes)
            
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image
        except Exception as e:
            if self.verbose:
                print(f"Error loading frame {frame_key}: {e}")
            # Return black frame as fallback
            if self.transform:
                black_frame = self.transform(Image.new('RGB', (294, 518)))
            else:
                black_frame = torch.zeros(3, 294, 518)
            return black_frame
    
    async def _download_single_frame_async(self, frame_key: str) -> torch.Tensor:
        """Download and process a single frame asynchronously"""
        try:
            # Use asyncio to run the sync S3 client call in a thread pool
            loop = asyncio.get_event_loop()
            
            # Download frame data in thread pool
            response = await loop.run_in_executor(
                None,
                lambda: self.s3_client.get_object(Bucket=self.bucket_name, Key=frame_key)
            )
            
            # Read bytes
            image_bytes = await loop.run_in_executor(
                None,
                response['Body'].read
            )
            
            # Process image (CPU-bound, so run in executor)
            image = await loop.run_in_executor(
                None, 
                lambda: Image.open(io.BytesIO(image_bytes)).convert('RGB')
            )
            
            if self.transform:
                image = await loop.run_in_executor(None, self.transform, image)
            
            return image
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading frame {frame_key}: {e}")
            # Return black frame as fallback
            if self.transform:
                black_frame = self.transform(Image.new('RGB', (294, 518)))
            else:
                black_frame = torch.zeros(3, 294, 518)
            return black_frame
    
    async def _load_frames_async_impl(self, frame_keys: List[str]) -> List[torch.Tensor]:
        """Load multiple frames asynchronously"""
        # Create tasks for all frames
        tasks = [
            self._download_single_frame_async(frame_key) 
            for frame_key in frame_keys
        ]
        
        # Download all frames concurrently
        frames = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions by replacing with black frames
        for i, frame in enumerate(frames):
            if isinstance(frame, Exception):
                if self.verbose:
                    print(f"Error in async download: {frame}")
                if self.transform:
                    frames[i] = self.transform(Image.new('RGB', (294, 518)))
                else:
                    frames[i] = torch.zeros(3, 294, 518)
        
        return frames
    
    def _load_frames_async(self, frame_keys: List[str]) -> List[torch.Tensor]:
        """Synchronous wrapper for async frame loading"""
        # Check if we're already in an event loop (common in Jupyter/DataLoader)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self._load_frames_async_impl(frame_keys))
        else:
            # Already in a loop, use nest_asyncio
            return asyncio.ensure_future(
                self._load_frames_async_impl(frame_keys)
            ).result()
    
    def _load_frames_parallel(self, frame_keys: List[str]) -> List[torch.Tensor]:
        """Load multiple frames in parallel"""
        # Use min of max_workers or number of frames to avoid over-threading
        num_workers = min(self.max_workers, len(frame_keys))
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all download tasks
            future_to_index = {
                executor.submit(self._download_single_frame, frame_key): i 
                for i, frame_key in enumerate(frame_keys)
            }
            
            # Initialize results list with correct size
            loaded_frames = [None] * len(frame_keys)
            
            # Collect results maintaining order
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    frame = future.result()
                    loaded_frames[index] = frame
                except Exception as e:
                    if self.verbose:
                        print(f"Error in parallel frame loading: {e}")
                    # Create fallback frame
                    if self.transform:
                        fallback_frame = self.transform(Image.new('RGB', (294, 518)))
                    else:
                        fallback_frame = torch.zeros(3, 294, 518)
                    loaded_frames[index] = fallback_frame
        
        return loaded_frames
    
    def _background_cache_worker(self):
        """Background thread that prefetches frames into RAM cache"""
        from queue import Empty
        
        while not self.cache_stop_event.is_set():
            try:
                # Get frame key to prefetch
                frame_key = self.prefetch_queue.get(timeout=1)
                if frame_key is None:  # Shutdown signal
                    break
                
                # Check if already in cache
                with self.cache_lock:
                    if frame_key in self.ram_cache:
                        continue
                
                # Download frame
                image_data = self._download_frame_for_cache(frame_key)
                if image_data:
                    self._add_to_ram_cache(frame_key, image_data)
                    
            except Empty:
                # Normal - queue is empty, continue waiting
                continue
            except Exception as e:
                if self.verbose:
                    print(f"Background cache error: {e}")
                time.sleep(0.1)
    
    def _download_frame_for_cache(self, frame_key: str) -> Optional[bytes]:
        """Download raw frame bytes from S3 for caching"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=frame_key)
            return response['Body'].read()
        except Exception as e:
            return None
    
    def _add_to_ram_cache(self, frame_key: str, image_data: bytes):
        """Add frame to RAM cache with LRU eviction"""
        with self.cache_lock:
            data_size = len(image_data)
            max_cache_bytes = self.max_cache_size_mb * 1024 * 1024
            
            # Evict old entries if needed
            while (self.current_cache_size + data_size > max_cache_bytes and 
                   len(self.ram_cache) > 0):
                self._evict_lru_frame()
            
            # Add new frame
            self.ram_cache[frame_key] = image_data
            self.current_cache_size += data_size
            self.cache_access_count[frame_key] = 0
    
    def _evict_lru_frame(self):
        """Evict least recently used frame from RAM cache"""
        if not self.ram_cache:
            return
        
        # Find frame with lowest access count
        lru_key = min(self.cache_access_count.keys(), 
                     key=lambda k: self.cache_access_count[k] if k in self.ram_cache else float('inf'))
        
        if lru_key in self.ram_cache:
            data_size = len(self.ram_cache[lru_key])
            del self.ram_cache[lru_key]
            del self.cache_access_count[lru_key]
            self.current_cache_size -= data_size
    
    def _get_frame_from_ram_cache(self, frame_key: str) -> Optional[bytes]:
        """Get frame from RAM cache and update access count"""
        with self.cache_lock:
            if frame_key in self.ram_cache:
                self.cache_access_count[frame_key] += 1
                return self.ram_cache[frame_key]
            return None
    
    def _queue_for_prefetch(self, frame_keys: List[str]):
        """Queue frames for background prefetching"""
        if not self.enable_background_cache:
            return  # Skip prefetching if disabled
        for frame_key in frame_keys:
            try:
                # Only queue if not already in cache
                with self.cache_lock:
                    if frame_key not in self.ram_cache:
                        self.prefetch_queue.put_nowait(frame_key)
            except:
                pass  # Queue full, skip
    
    def _predict_next_frames(self, current_seq: Dict, local_idx: int) -> List[str]:
        """Predict next frames that might be accessed"""
        frames = current_seq['frames']
        next_frames = []
        
        # Prefetch next few sequences from same video
        for i in range(1, 4):  # Next 3 sequences
            next_start = local_idx + (i * self.total_frames)
            if next_start + self.total_frames <= len(frames):
                sequence_frames = frames[next_start:next_start + self.total_frames]
                next_frames.extend(sequence_frames)
        
        return next_frames
    
    def _batch_download_frames(self, frame_keys: List[str]) -> Dict[str, bytes]:
        """Download multiple frames in parallel"""
        downloaded = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_key = {
                executor.submit(self._download_frame, key): key 
                for key in frame_keys
            }
            
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    image_data = future.result()
                    if image_data:
                        downloaded[key] = image_data
                except Exception as e:
                    print(f"Error downloading {key}: {e}")
        
        return downloaded
    
    def _get_channel_stats(self) -> Dict[str, int]:
        """Get statistics per channel"""
        stats = defaultdict(int)
        for seq in self.sequences:
            stats[seq['channel']] += 1
        return dict(stats)
    
    def _get_video_stats(self) -> Dict[str, int]:
        """Get statistics per video"""
        stats = defaultdict(int)
        for seq in self.sequences:
            video_key = f"{seq['channel']}/{seq['video']}"
            stats[video_key] = len(seq['frames'])
        return dict(stats)
    
    def __len__(self) -> int:
        """Total number of possible sequences accounting for frame sampling"""
        total = 0
        for seq in self.sequences:
            # Account for frame sampling rate
            frames_needed = (self.total_frames - 1) * self.frame_sampling_rate + 1
            available_frames = len(seq['frames'])
            if available_frames >= frames_needed:
                # Number of valid starting positions when sampling
                valid_positions = available_frames - frames_needed + 1
                total += valid_positions
        return total
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get a sequence of consecutive frames.
        
        Returns:
            current_frames: Tensor of shape [m, C, H, W]
            future_frames: Tensor of shape [n, C, H, W]
            metadata: Dict with channel, video, and frame info
        """
        # Find which sequence this index belongs to
        current_idx = 0
        for seq in self.sequences:
            # Account for frame sampling when calculating sequence length
            frames_needed = (self.total_frames - 1) * self.frame_sampling_rate + 1
            available_frames = len(seq['frames'])
            
            if available_frames >= frames_needed:
                valid_positions = available_frames - frames_needed + 1
                
                if idx < current_idx + valid_positions:
                    # Found the sequence
                    local_idx = idx - current_idx
                    
                    # Sample frames at the specified rate
                    frame_indices = [local_idx + i * self.frame_sampling_rate 
                                   for i in range(self.total_frames)]
                    frames_to_load = [seq['frames'][idx] for idx in frame_indices]
                    
                    # Queue next frames for background prefetching
                    # next_frames = self._predict_next_frames(seq, local_idx)
                    # if next_frames:
                    #     self._queue_for_prefetch(next_frames)
                    
                    # Load frames in parallel or async
                    if self.use_async:
                        loaded_frames = self._load_frames_async(frames_to_load)
                    else:
                        loaded_frames = self._load_frames_parallel(frames_to_load)
                    
                    # Stack frames
                    # go through every frame and make sure they are size 294 by 518
                    expected_shape = (3, 294, 518)  # (C, H, W)
                    
                    for i, frame in enumerate(loaded_frames):
                        if frame.shape != expected_shape:
                            print(f"Warning: Frame {i} has shape {frame.shape}, expected {expected_shape}")
                            # Resize to correct shape if needed
                            if frame.dim() == 3 and frame.shape[0] == 3:
                                # Interpolate to correct height and width
                                frame = torch.nn.functional.interpolate(
                                    frame.unsqueeze(0),  # Add batch dimension
                                    size=(294, 518),  # (H, W)
                                    mode='bilinear',
                                    align_corners=False
                                ).squeeze(0)  # Remove batch dimension
                                loaded_frames[i] = frame
                            else:
                                # If completely wrong shape, replace with black frame
                                print(f"Error: Frame {i} has invalid shape {frame.shape}, replacing with black frame")
                                loaded_frames[i] = torch.zeros(expected_shape)
                    
                    all_frames = torch.stack(loaded_frames)
                    current_frames = all_frames[:self.m]
                    future_frames = all_frames[self.m:]
                    
                    # Metadata
                    metadata = {
                        'channel': seq['channel'],
                        'video': seq['video'],
                        'start_frame_idx': local_idx,
                        'frame_paths': frames_to_load,
                        'frame_sampling_rate': self.frame_sampling_rate,
                        'effective_fps': 10.0 / self.frame_sampling_rate  # Original is 10Hz
                    }
                    
                    return current_frames, future_frames, metadata
                
                current_idx += valid_positions
            else:
                # Skip sequences that don't have enough frames for sampling
                continue
        
        raise IndexError(f"Index {idx} out of range for dataset with size {len(self)}")
    
    def get_random_sequence(self) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Get a random sequence - useful for testing"""
        import random
        idx = random.randint(0, len(self) - 1)
        return self[idx]
    
    def __del__(self):
        """Cleanup background threads"""
        self.cache_stop_event.set()
        try:
            self.prefetch_queue.put_nowait(None)  # Signal shutdown
        except:
            pass
        if hasattr(self, 'background_cache_thread') and self.background_cache_thread:
            self.background_cache_thread.join(timeout=1)
    
    def get_cache_stats(self):
        """Get current cache statistics"""
        with self.cache_lock:
            return {
                'cache_size_mb': self.current_cache_size / (1024*1024),
                'max_size_mb': self.max_cache_size_mb,
                'cached_frames': len(self.ram_cache),
                'utilization': (self.current_cache_size / (self.max_cache_size_mb * 1024 * 1024)) * 100
            }
    
    def print_dataset_info(self):
        """Print detailed dataset information"""
        print("\n" + "="*60)
        print("YouTube S3 Dataset Detailed Information")
        print("="*60)
        
        # Channel stats
        channel_stats = self._get_channel_stats()
        print(f"\n📺 Channels ({len(channel_stats)} total):")
        for channel, count in sorted(channel_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   - {channel}: {count} videos")
        
        if len(channel_stats) > 10:
            print(f"   ... and {len(channel_stats) - 10} more channels")
        
        # Video stats
        video_stats = self._get_video_stats()
        sorted_videos = sorted(video_stats.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🎬 Top 10 Longest Videos:")
        for video, frame_count in sorted_videos[:10]:
            print(f"   - {video}: {frame_count} frames")
        
        # Overall stats
        print(f"\n📊 Overall Statistics:")
        print(f"   - Total sequences: {len(self)}")
        print(f"   - Average frames per video: {sum(video_stats.values()) / len(video_stats):.1f}")
        print(f"   - Total frames: {sum(video_stats.values())}")
        print(f"   - Sequence length (M+N): {self.total_frames}")
        print(f"   - Frame sampling rate: {self.frame_sampling_rate} (every {self.frame_sampling_rate} frames)")
        print(f"   - Effective FPS: {10.0/self.frame_sampling_rate:.1f} Hz (from 10Hz source)")
        print(f"   - Temporal span per sequence: {(self.total_frames - 1) * self.frame_sampling_rate / 10.0:.1f}s")
        print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Create dataset
    dataset = YouTubeS3Dataset(
        bucket_name="research-datasets",
        root_prefix="openDV-YouTube/full_images/",
        m=10,
        n=0,
        cache_dir="./youtube_cache",
        refresh_cache=False,  # Set to True to rebuild cache
        skip_frames=300,  # Skip first 300 frames like the original dataset
        verbose=True
    )
    
    # Print dataset info
    dataset.print_dataset_info()
    
    # Test loading a sequence
    print("\n🧪 Testing data loading...")
    current, future, metadata = dataset.get_random_sequence()
    print(f"   Loaded sequence from: {metadata['channel']}/{metadata['video']}")
    print(f"   Current frames shape: {current.shape}")
    print(f"   Future frames shape: {future.shape}")


# class YouTubeShardDatasetWrapper:
#     """
#     Wrapper to make YouTubeVideoDataset (IterableDataset) compatible with existing training code.
#     Handles frame preprocessing, target generation, and batching.
#     """
    
#     def __init__(
#         self,
#         s3_path: str,
#         m: int = 3,
#         n: int = 3,
#         target_width: int = 518,
#         target_height: int = 294,
#         max_shards: Optional[int] = None,
#         transform: Optional[T.Compose] = None
#     ):
#         """
#         Args:
#             s3_path: S3 path to sharded dataset
#             m: Number of input frames
#             n: Number of target frames
#             target_width: Target width for frames
#             target_height: Target height for frames
#             max_shards: Optional limit on number of shards
#             transform: Optional transform to apply to frames
#         """
#         self.m = m
#         self.n = n
#         self.target_width = target_width
#         self.target_height = target_height
#         self.total_frames = m + n
        
#         # Create the underlying iterable dataset
#         self.dataset = YouTubeVideoDataset(s3_path=s3_path, max_shards=max_shards)
        
#         # Default transform if none provided
#         self.transform = transform or T.Compose([
#             T.Resize((target_height, target_width)),
#             T.ToTensor(),
#         ])
    
#     def _preprocess_frames(self, frames: torch.Tensor) -> torch.Tensor:
#         """
#         Preprocess frames from uint8 [seq_len, H, W, C] to float32 [seq_len, C, H, W].
        
#         Args:
#             frames: Tensor of shape [seq_len, H, W, C] with dtype uint8
            
#         Returns:
#             Tensor of shape [seq_len, C, H, W] with dtype float32, normalized to [0, 1]
#         """
#         # Convert to float and normalize to [0, 1]
#         frames_float = frames.float() / 255.0
        
#         # Rearrange from [seq_len, H, W, C] to [seq_len, C, H, W]
#         frames_processed = frames_float.permute(0, 3, 1, 2)
        
#         # Resize to target dimensions if needed
#         seq_len, C, H, W = frames_processed.shape
#         if H != self.target_height or W != self.target_width:
#             frames_processed = F.interpolate(
#                 frames_processed,
#                 size=(self.target_height, self.target_width),
#                 mode='bilinear',
#                 align_corners=False
#             )
        
#         return frames_processed
    
#     def _create_sliding_window_sequences(self, frames: torch.Tensor, metadata: Dict) -> List[Tuple]:
#         """
#         Create multiple overlapping sequences from a video using sliding window.
        
#         Args:
#             frames: Preprocessed frames [seq_len, C, H, W]
#             metadata: Metadata dict for the video
            
#         Returns:
#             List of (current_frames, future_frames, metadata) tuples
#         """
#         sequences = []
#         seq_len = frames.shape[0]
        
#         # Create sliding window sequences
#         for start_idx in range(seq_len - self.total_frames + 1):
#             window_frames = frames[start_idx:start_idx + self.total_frames]
            
#             current_frames = window_frames[:self.m]
#             future_frames = window_frames[self.m:]
            
#             # Create metadata for this sequence
#             seq_metadata = {
#                 **metadata,
#                 'start_frame_idx': start_idx,
#                 'sequence_length': self.total_frames
#             }
            
#             sequences.append((current_frames, future_frames, seq_metadata))
        
#         return sequences
    
#     def __iter__(self):
#         """
#         Iterate through all sequences in all shards.
#         Yields (current_frames, future_frames, metadata) tuples.
#         """
#         for sample in self.dataset:
#             frames = sample['frames']  # [seq_len, H, W, C] uint8
#             metadata = sample['metadata']
            
#             # Skip videos that are too short
#             if frames.shape[0] < self.total_frames:
#                 continue
            
#             # Preprocess frames
#             processed_frames = self._preprocess_frames(frames)
            
#             # Create sliding window sequences
#             sequences = self._create_sliding_window_sequences(processed_frames, metadata)
            
#             # Yield each sequence
#             for sequence in sequences:
#                 yield sequence
    
#     def __repr__(self) -> str:
#         return f"YouTubeShardDatasetWrapper(m={self.m}, n={self.n}, target_size=({self.target_height}, {self.target_width}), dataset={self.dataset})"