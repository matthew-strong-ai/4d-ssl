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


class YouTubeS3DatasetOptimized(Dataset):
    """
    Optimized S3 dataset for YouTube channel structure with batch downloading and caching.
    
    Key optimizations:
    1. Batch downloading of frames instead of one-by-one
    2. Local disk caching of downloaded frames
    3. In-memory LRU cache for hot frames
    4. Background prefetching of upcoming sequences
    5. Parallel downloads with connection pooling
    
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
        max_workers: int = 32,  # Increased for parallel S3 operations
        verbose: bool = True,
        batch_size: int = 100,  # Download frames in batches
        prefetch_sequences: int = 10,  # Number of sequences to prefetch
        memory_cache_size: int = 10000,  # Max frames to keep in memory
        enable_local_cache: bool = True  # Enable disk caching
    ):
        """
        Args:
            bucket_name: S3 bucket name
            root_prefix: Root prefix for YouTube dataset
            m: Number of input frames
            n: Number of target frames  
            transform: Optional transforms to apply to images
            region_name: AWS region
            cache_dir: Directory to cache file listings and frames
            refresh_cache: If True, refresh the cache even if it exists
            min_sequence_length: Minimum video length to include
            skip_frames: Number of frames to skip at the beginning of each video
            max_workers: Number of parallel workers for S3 operations
            verbose: Print detailed progress
            batch_size: Number of frames to download in a single batch
            prefetch_sequences: Number of sequences to prefetch in background
            memory_cache_size: Maximum number of frames to keep in memory
            enable_local_cache: Whether to cache frames on disk
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
        self.min_sequence_length = max(min_sequence_length, self.total_frames + skip_frames)
        self.skip_frames = skip_frames
        self.max_workers = max_workers
        self.verbose = verbose
        self.batch_size = batch_size
        self.prefetch_sequences = prefetch_sequences
        self.memory_cache_size = memory_cache_size
        self.enable_local_cache = enable_local_cache
        
        # Initialize S3 client with better configuration
        # Get endpoint URL from environment if available
        endpoint_url = os.environ.get('AWS_ENDPOINT_URL')
        
        client_kwargs = {
            'region_name': 'us-phoenix-1',  # Use Oracle Cloud region
            'config': Config(
                max_pool_connections=max(50, max_workers * 2),  # Increased pool size
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
        
        # Initialize caching systems
        self.frame_cache = {}  # In-memory cache {frame_key: image_bytes}
        self.cache_access_times = {}  # Track access times for LRU
        self.cache_lock = threading.Lock()
        
        # Local disk cache directory
        if self.enable_local_cache:
            self.local_cache_dir = self.cache_dir / "frame_cache"
            self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.local_cache_dir = None
        
        # Prefetch queue and background thread
        self.prefetch_queue = Queue(maxsize=self.prefetch_sequences * 2)
        self.download_threads = []
        self.stop_threads = threading.Event()
        
        # Build or load the dataset index
        self.sequences = self._build_dataset_index(refresh_cache)
        
        # Start background download threads
        for i in range(min(4, max_workers // 4)):  # Use 1/4 of workers for background prefetch
            thread = threading.Thread(target=self._background_downloader, daemon=True)
            thread.start()
            self.download_threads.append(thread)
        
        if self.verbose:
            print(f"\n📊 YouTube S3 Dataset Statistics (Optimized):")
            print(f"   Total channels: {len(self._get_channel_stats())}")
            print(f"   Total videos: {len(self._get_video_stats())}")
            print(f"   Total valid sequences: {len(self.sequences)}")
            print(f"   Total frames available: {sum(len(seq['frames']) for seq in self.sequences)}")
            print(f"   Batch download size: {self.batch_size}")
            print(f"   Prefetch sequences: {self.prefetch_sequences}")
            print(f"   Memory cache size: {self.memory_cache_size}")
            print(f"   Local disk cache: {'Enabled' if self.enable_local_cache else 'Disabled'}")
            print(f"   Background threads: {len(self.download_threads)}")
    
    def __del__(self):
        """Cleanup background threads"""
        self.stop_threads.set()
        for thread in self.download_threads:
            thread.join(timeout=1)
    
    def _default_transform(self):
        """Default transform if none provided"""
        return T.Compose([
            T.Resize((294, 518)),
            T.ToTensor(),
        ])
    
    def _get_cache_path(self) -> Path:
        """Generate cache file path based on dataset parameters"""
        cache_key = hashlib.md5(
            f"{self.bucket_name}_{self.root_prefix}_{self.min_sequence_length}_{self.skip_frames}".encode()
        ).hexdigest()
        return self.cache_dir / f"youtube_dataset_{cache_key}.pkl"
    
    def _build_dataset_index(self, refresh_cache: bool) -> List[Dict]:
        """Build or load the dataset index"""
        cache_path = self._get_cache_path()
        
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
            
            if len(frames) >= self.min_sequence_length:
                # Skip first N frames and create sequence entry
                valid_frames = frames[self.skip_frames:]
                if len(valid_frames) >= self.total_frames:
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
        while not self.stop_threads.is_set():
            try:
                # Get batch of frames to download
                batch_request = self.prefetch_queue.get(timeout=1)
                if batch_request is None:  # Shutdown signal
                    break
                
                frame_keys = batch_request
                
                # Download frames in batch
                downloaded = self._batch_download_frames(frame_keys)
                
                # Add to cache
                with self.cache_lock:
                    for key, data in downloaded.items():
                        self._add_to_cache(key, data)
                
            except:
                continue
    
    def _add_to_cache(self, key: str, data: bytes):
        """Add frame to cache with LRU eviction"""
        # Update access time
        self.cache_access_times[key] = time.time()
        self.frame_cache[key] = data
        
        # Evict old entries if cache is too large
        if len(self.frame_cache) > self.memory_cache_size:
            # Find oldest entries
            sorted_keys = sorted(self.cache_access_times.items(), key=lambda x: x[1])
            keys_to_remove = [k for k, _ in sorted_keys[:len(self.frame_cache) - self.memory_cache_size]]
            
            for key in keys_to_remove:
                self.frame_cache.pop(key, None)
                self.cache_access_times.pop(key, None)
    
    def _get_local_cache_path(self, frame_key: str) -> Path:
        """Get local cache file path for a frame"""
        # Create a safe filename from the S3 key
        safe_name = frame_key.replace('/', '_').replace('\\', '_')
        return self.local_cache_dir / safe_name
    
    def _download_frame(self, frame_key: str) -> Optional[bytes]:
        """Download a single frame from S3 and optionally cache locally"""
        # Check local disk cache first
        if self.enable_local_cache:
            local_path = self._get_local_cache_path(frame_key)
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
            if self.enable_local_cache:
                try:
                    with open(local_path, 'wb') as f:
                        f.write(image_bytes)
                except:
                    pass  # Ignore cache write errors
            
            return image_bytes
        except Exception as e:
            if self.verbose:
                print(f"Error downloading frame {frame_key}: {e}")
            return None
    
    def _batch_download_frames(self, frame_keys: List[str]) -> Dict[str, bytes]:
        """Download multiple frames in parallel"""
        downloaded = {}
        frames_to_download = []
        
        # Check local cache first
        if self.enable_local_cache:
            for key in frame_keys:
                local_path = self._get_local_cache_path(key)
                if local_path.exists():
                    try:
                        with open(local_path, 'rb') as f:
                            downloaded[key] = f.read()
                    except:
                        frames_to_download.append(key)
                else:
                    frames_to_download.append(key)
        else:
            frames_to_download = frame_keys
        
        # Download missing frames in parallel
        if frames_to_download:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(frames_to_download))) as executor:
                future_to_key = {
                    executor.submit(self._download_frame, key): key 
                    for key in frames_to_download
                }
                
                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        image_data = future.result()
                        if image_data:
                            downloaded[key] = image_data
                    except Exception as e:
                        if self.verbose:
                            print(f"Error in batch download for {key}: {e}")
        
        return downloaded
    
    def _prefetch_sequences(self, current_seq: Dict, current_local_idx: int):
        """Prefetch upcoming sequences in the background"""
        try:
            # Prefetch frames from current video
            remaining_in_video = len(current_seq['frames']) - current_local_idx - self.total_frames
            
            if remaining_in_video > 0:
                # Prefetch next few sequences from current video
                for i in range(min(self.prefetch_sequences, remaining_in_video)):
                    next_idx = current_local_idx + self.total_frames + i * self.total_frames
                    if next_idx + self.total_frames <= len(current_seq['frames']):
                        frames_to_prefetch = current_seq['frames'][next_idx:next_idx + self.total_frames]
                        
                        # Check if already in cache
                        frames_needed = []
                        with self.cache_lock:
                            for frame in frames_to_prefetch:
                                if frame not in self.frame_cache:
                                    frames_needed.append(frame)
                        
                        if frames_needed and not self.prefetch_queue.full():
                            self.prefetch_queue.put_nowait(frames_needed)
        except:
            pass  # Ignore prefetch errors
    
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
        """Total number of possible consecutive sequences"""
        total = 0
        for seq in self.sequences:
            total += len(seq['frames']) - self.total_frames + 1
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
            seq_length = len(seq['frames']) - self.total_frames + 1
            if idx < current_idx + seq_length:
                # Found the sequence
                local_idx = idx - current_idx
                frames_to_load = seq['frames'][local_idx:local_idx + self.total_frames]
                
                # Prefetch future sequences
                self._prefetch_sequences(seq, local_idx)
                
                # Check cache first
                frames_needed = []
                cached_frames = {}
                
                with self.cache_lock:
                    for frame_key in frames_to_load:
                        if frame_key in self.frame_cache:
                            cached_frames[frame_key] = self.frame_cache[frame_key]
                            # Update access time
                            self.cache_access_times[frame_key] = time.time()
                        else:
                            frames_needed.append(frame_key)
                
                # Batch download any missing frames
                if frames_needed:
                    downloaded = self._batch_download_frames(frames_needed)
                    
                    # Add to cache
                    with self.cache_lock:
                        for key, data in downloaded.items():
                            self._add_to_cache(key, data)
                    
                    cached_frames.update(downloaded)
                
                # Load frames
                loaded_frames = []
                for frame_key in frames_to_load:
                    try:
                        if frame_key in cached_frames:
                            image_bytes = cached_frames[frame_key]
                        else:
                            # Fallback to direct download if not in cache
                            image_bytes = self._download_frame(frame_key)
                            if image_bytes:
                                with self.cache_lock:
                                    self._add_to_cache(frame_key, image_bytes)
                        
                        if image_bytes:
                            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                            if self.transform:
                                image = self.transform(image)
                            loaded_frames.append(image)
                        else:
                            raise Exception("Failed to load frame")
                            
                    except Exception as e:
                        if self.verbose:
                            print(f"Error loading frame {frame_key}: {e}")
                        # Return black frame as fallback
                        if self.transform:
                            black_frame = self.transform(Image.new('RGB', (294, 518)))
                        else:
                            black_frame = torch.zeros(3, 294, 518)
                        loaded_frames.append(black_frame)
                
                # Stack frames
                # Ensure all frames have the correct shape
                expected_shape = (3, 294, 518)  # (C, H, W)
                
                for i, frame in enumerate(loaded_frames):
                    if frame.shape != expected_shape:
                        if self.verbose:
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
                            if self.verbose:
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
                    'frame_paths': frames_to_load
                }
                
                return current_frames, future_frames, metadata
            
            current_idx += seq_length
        
        raise IndexError(f"Index {idx} out of range for dataset with size {len(self)}")
    
    def get_random_sequence(self) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Get a random sequence - useful for testing"""
        import random
        idx = random.randint(0, len(self) - 1)
        return self[idx]
    
    def print_dataset_info(self):
        """Print detailed dataset information"""
        print("\n" + "="*60)
        print("YouTube S3 Dataset Detailed Information (Optimized)")
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
        
        # Cache stats
        with self.cache_lock:
            print(f"\n💾 Cache Statistics:")
            print(f"   - Frames in memory cache: {len(self.frame_cache)}")
            print(f"   - Memory cache capacity: {self.memory_cache_size}")
            print(f"   - Local disk cache: {'Enabled' if self.enable_local_cache else 'Disabled'}")
            if self.enable_local_cache and self.local_cache_dir.exists():
                cache_files = list(self.local_cache_dir.glob('*'))
                print(f"   - Files in disk cache: {len(cache_files)}")
        
        print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Create optimized dataset
    dataset = YouTubeS3DatasetOptimized(
        bucket_name="research-datasets",
        root_prefix="openDV-YouTube/full_images/",
        m=3,
        n=3,
        cache_dir="./youtube_cache",
        refresh_cache=False,  # Set to True to rebuild cache
        skip_frames=300,  # Skip first 300 frames like the original dataset
        max_workers=32,  # Increased parallel workers
        batch_size=100,  # Download 100 frames at a time
        prefetch_sequences=10,  # Prefetch 10 sequences ahead
        memory_cache_size=10000,  # Keep 10k frames in memory
        enable_local_cache=True,  # Enable disk caching
        verbose=True
    )
    
    # Print dataset info
    dataset.print_dataset_info()
    
    # Test loading a sequence
    print("\n🧪 Testing data loading...")
    import time
    start_time = time.time()
    current, future, metadata = dataset.get_random_sequence()
    load_time = time.time() - start_time
    print(f"   Loaded sequence from: {metadata['channel']}/{metadata['video']}")
    print(f"   Current frames shape: {current.shape}")
    print(f"   Future frames shape: {future.shape}")
    print(f"   Load time: {load_time:.2f} seconds")
    
    # Test loading multiple sequences to see cache benefit
    print("\n⚡ Testing cache performance...")
    # Load same sequence again
    start_time = time.time()
    current2, future2, metadata2 = dataset[0]
    cached_time = time.time() - start_time
    print(f"   First load time: {load_time:.2f} seconds")
    print(f"   Cached load time: {cached_time:.2f} seconds")
    print(f"   Speedup: {load_time/cached_time:.1f}x")