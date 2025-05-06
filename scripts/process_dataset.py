#!/usr/bin/env python
"""
Dataset Processing Script for Self-Guiding Sentinels

This script processes the UHCTD dataset and prepares it for training. It:
1. Extracts frames from videos
2. Normalizes file organization
3. Creates train/validation/test splits
4. Generates metadata
5. Creates cross-camera validation files
6. Integrates synthetic data (optional)

Usage:
    python scripts/process_dataset.py --uhctd-dir /path/to/uhctd --output-dir data/processed
    python scripts/process_dataset.py --uhctd-dir /path/to/uhctd --output-dir data/processed --synthetic-dir data/synthetic
"""

import os
import sys
import argparse
import shutil
import pandas as pd
import numpy as np
import cv2
import json
import random
import logging
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import glob
from typing import List, Dict, Tuple, Optional, Union, Any
import re
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('process_dataset.log')
    ]
)
logger = logging.getLogger(__name__)


class DatasetProcessor:
    """
    Class for processing the UHCTD dataset and preparing it for training.
    """
    def __init__(self, args):
        """
        Initialize the dataset processor.
        
        Args:
            args: Command-line arguments
        """
        self.args = args
        self.uhctd_dir = args.uhctd_dir
        self.output_dir = args.output_dir
        self.synthetic_dir = args.synthetic_dir
        self.splits_ratio = args.splits_ratio
        self.extract_frames = args.extract_frames
        self.frame_interval = args.frame_interval
        self.create_metadata = args.create_metadata
        self.num_workers = args.num_workers
        
        # Setup output directories
        self._setup_directories()
        
        # Load existing annotations if available
        self._load_annotations()
    
    def _setup_directories(self):
        """Create output directory structure."""
        # Main output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Subdirectories
        self.videos_dir = os.path.join(self.output_dir, "videos")
        self.frames_dir = os.path.join(self.output_dir, "frames")
        self.metadata_dir = os.path.join(self.output_dir, "metadata")
        self.annotations_dir = os.path.join(self.output_dir, "annotations")
        
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
        
        # Create splits directories
        if self.splits_ratio:
            os.makedirs(os.path.join(self.output_dir, "train"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "val"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "test"), exist_ok=True)
    
    def _load_annotations(self):
        """Load UHCTD annotations if available."""
        # Try to find the main annotation file
        potential_paths = [
            os.path.join(self.uhctd_dir, "annotations.csv"),
            os.path.join(self.uhctd_dir, "metadata", "annotations.csv"),
            os.path.join(self.uhctd_dir, "UHCTD_annotations.csv")
        ]
        
        self.annotations = None
        
        for path in potential_paths:
            if os.path.exists(path):
                try:
                    self.annotations = pd.read_csv(path)
                    logger.info(f"Loaded annotations from {path} with {len(self.annotations)} entries")
                    break
                except Exception as e:
                    logger.error(f"Error loading annotations from {path}: {str(e)}")
        
        if self.annotations is None:
            logger.warning("No annotations found. Will try to create them from video filenames.")
    
    def process(self):
        """Process the dataset."""
        # Process the UHCTD dataset
        self._process_uhctd()
        
        # Process synthetic data if provided
        if self.synthetic_dir:
            self._process_synthetic_data()
        
        # Create dataset splits
        if self.splits_ratio:
            self._create_splits()
        
        # Create metadata
        if self.create_metadata:
            self._create_metadata()
        
        logger.info("Dataset processing complete!")
    
    def _process_uhctd(self):
        """Process the UHCTD dataset."""
        logger.info("Processing UHCTD dataset...")
        
        # Find videos
        video_files = self._find_video_files(self.uhctd_dir)
        
        if not video_files:
            logger.error(f"No video files found in {self.uhctd_dir}")
            return
        
        logger.info(f"Found {len(video_files)} video files")
        
        # Create annotations if not available
        if self.annotations is None:
            self.annotations = self._create_annotations_from_filenames(video_files)
        
        # Copy videos and extract frames
        self._process_videos(video_files)
        
        # Save processed annotations
        self._save_annotations()
    
    def _find_video_files(self, directory):
        """
        Find all video files in the directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of video file paths
        """
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(directory, "**", f"*{ext}"), recursive=True))
        
        return video_files
    
    def _create_annotations_from_filenames(self, video_files):
        """
        Create annotations dataframe from video filenames.
        
        Args:
            video_files: List of video file paths
            
        Returns:
            pandas.DataFrame: Annotations dataframe
        """
        logger.info("Creating annotations from filenames...")
        
        annotations = []
        
        for video_path in video_files:
            filename = os.path.basename(video_path)
            
            # Try to extract information from filename
            try:
                # Assume filename format: [type]_[scene]_[timestamp].mp4
                # Example: attack_office_20210315.mp4
                parts = os.path.splitext(filename)[0].split('_')
                
                if len(parts) >= 1:
                    event_type = parts[0].lower()
                    # Map to standard class names
                    if 'attack' in event_type or 'tamper' in event_type:
                        event_type = 'attack'
                    elif 'normal' in event_type:
                        event_type = 'normal'
                    elif 'environment' in event_type:
                        event_type = 'environmental'
                    elif 'maintenance' in event_type:
                        event_type = 'maintenance'
                    else:
                        event_type = 'unknown'
                else:
                    event_type = 'unknown'
                
                # Get video properties
                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                
                # Create annotation entry
                annotations.append({
                    'video_name': filename,
                    'event_type': event_type,
                    'start_frame': 0,
                    'end_frame': frame_count,
                    'fps': fps,
                    'duration': frame_count / fps if fps > 0 else 0
                })
                
            except Exception as e:
                logger.warning(f"Error processing filename {filename}: {str(e)}")
                continue
        
        # Create dataframe
        annotations_df = pd.DataFrame(annotations)
        logger.info(f"Created annotations for {len(annotations_df)} videos")
        
        return annotations_df
    
    def _process_videos(self, video_files):
        """
        Process videos: copy to output dir and extract frames if needed.
        
        Args:
            video_files: List of video file paths
        """
        logger.info("Processing videos...")
        
        # Create a mapping of video filename to path
        video_path_map = {os.path.basename(v): v for v in video_files}
        
        tasks = []
        
        # Process each video in the annotations
        for _, row in self.annotations.iterrows():
            video_name = row['video_name']
            
            # Find the video path
            video_path = video_path_map.get(video_name)
            if not video_path and 'video_path' in row:
                video_path = row['video_path']
            
            if not video_path or not os.path.exists(video_path):
                logger.warning(f"Video file not found for {video_name}")
                continue
            
            # Destination path
            dest_path = os.path.join(self.videos_dir, video_name)
            
            # Copy video if it doesn't exist in the output dir
            if not os.path.exists(dest_path):
                try:
                    shutil.copy2(video_path, dest_path)
                    logger.debug(f"Copied {video_name} to {dest_path}")
                except Exception as e:
                    logger.error(f"Error copying {video_name}: {str(e)}")
                    continue
            
            if self.extract_frames:
                # Determine frame extraction parameters
                start_frame = int(row.get('start_frame', 0))
                end_frame = int(row.get('end_frame', float('inf')))
                
                # Create a task for frame extraction
                tasks.append((dest_path, video_name, start_frame, end_frame, self.frame_interval))
        
        # Extract frames in parallel if needed
        if self.extract_frames and tasks:
            self._extract_frames_parallel(tasks)
    
    def _extract_frames_parallel(self, tasks):
        """
        Extract frames from videos in parallel.
        
        Args:
            tasks: List of tuples (video_path, video_name, start_frame, end_frame, interval)
        """
        logger.info(f"Extracting frames using {self.num_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._extract_frames_from_video, *task) for task in tasks]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting frames"):
                try:
                    video_name, num_frames = future.result()
                    logger.debug(f"Extracted {num_frames} frames from {video_name}")
                except Exception as e:
                    logger.error(f"Error extracting frames: {str(e)}")
    
    def _extract_frames_from_video(self, video_path, video_name, start_frame, end_frame, interval):
        """
        Extract frames from a video.
        
        Args:
            video_path: Path to video file
            video_name: Video filename
            start_frame: Starting frame index
            end_frame: Ending frame index
            interval: Frame interval
            
        Returns:
            Tuple of (video_name, number of frames extracted)
        """
        try:
            # Create output directory for frames
            video_basename = os.path.splitext(video_name)[0]
            frames_output_dir = os.path.join(self.frames_dir, video_basename)
            os.makedirs(frames_output_dir, exist_ok=True)
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Could not open video {video_path}")
                return video_name, 0
            
            # Get frame count if end_frame is not specified
            if end_frame == float('inf'):
                end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Extract frames
            count = 0
            frame_idx = start_frame
            
            while frame_idx < end_frame:
                # Set position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                # Read frame
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Save frame
                frame_path = os.path.join(frames_output_dir, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                
                # Move to next frame
                frame_idx += interval
                count += 1
            
            cap.release()
            return video_name, count
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {str(e)}")
            return video_name, 0
    
    def _save_annotations(self):
        """Save processed annotations."""
        # Save full annotations
        annotations_path = os.path.join(self.annotations_dir, "annotations.csv")
        self.annotations.to_csv(annotations_path, index=False)
        logger.info(f"Saved annotations to {annotations_path}")
        
        # Save class-specific annotations
        for event_type in self.annotations['event_type'].unique():
            class_df = self.annotations[self.annotations['event_type'] == event_type]
            class_path = os.path.join(self.annotations_dir, f"{event_type}_annotations.csv")
            class_df.to_csv(class_path, index=False)
            logger.info(f"Saved {event_type} annotations with {len(class_df)} entries")
    
    def _process_synthetic_data(self):
        """Process synthetic data."""
        if not self.synthetic_dir or not os.path.exists(self.synthetic_dir):
            logger.warning(f"Synthetic data directory {self.synthetic_dir} not found")
            return
        
        logger.info(f"Processing synthetic data from {self.synthetic_dir}")
        
        # Look for synthetic annotations
        synthetic_annotations_path = os.path.join(self.synthetic_dir, "synthetic_annotations.csv")
        
        if not os.path.exists(synthetic_annotations_path):
            logger.warning(f"Synthetic annotations not found at {synthetic_annotations_path}")
            return
        
        try:
            # Load synthetic annotations
            synthetic_annotations = pd.read_csv(synthetic_annotations_path)
            logger.info(f"Loaded {len(synthetic_annotations)} synthetic annotations")
            
            # Add a column to indicate synthetic data
            synthetic_annotations['synthetic'] = True
            
            # Copy synthetic videos
            synthetic_videos_dir = os.path.join(self.synthetic_dir, "videos")
            if os.path.exists(synthetic_videos_dir):
                for _, row in synthetic_annotations.iterrows():
                    video_name = row['video_name']
                    src_path = os.path.join(synthetic_videos_dir, video_name)
                    dest_path = os.path.join(self.videos_dir, f"synthetic_{video_name}")
                    
                    if os.path.exists(src_path) and not os.path.exists(dest_path):
                        shutil.copy2(src_path, dest_path)
                        # Update the video name to reflect the new filename
                        synthetic_annotations.loc[synthetic_annotations['video_name'] == video_name, 'video_name'] = f"synthetic_{video_name}"
            
            # Copy synthetic frames if available
            synthetic_frames_dir = os.path.join(self.synthetic_dir, "frames")
            if os.path.exists(synthetic_frames_dir) and self.extract_frames:
                for subdir in os.listdir(synthetic_frames_dir):
                    src_frames_dir = os.path.join(synthetic_frames_dir, subdir)
                    if os.path.isdir(src_frames_dir):
                        dest_frames_dir = os.path.join(self.frames_dir, f"synthetic_{subdir}")
                        if not os.path.exists(dest_frames_dir):
                            shutil.copytree(src_frames_dir, dest_frames_dir)
            
            # Merge with original annotations
            if 'synthetic' not in self.annotations.columns:
                self.annotations['synthetic'] = False
            
            self.annotations = pd.concat([self.annotations, synthetic_annotations], ignore_index=True)
            logger.info(f"Merged synthetic data, new annotation count: {len(self.annotations)}")
            
            # Save updated annotations
            self._save_annotations()
            
        except Exception as e:
            logger.error(f"Error processing synthetic data: {str(e)}")
    
    def _create_splits(self):
        """Create train/validation/test splits."""
        logger.info("Creating dataset splits...")
        
        try:
            # Parse splits ratio
            splits = list(map(float, self.splits_ratio.split(':')))
            if len(splits) != 3:
                logger.error("Splits ratio must have 3 values (train:val:test)")
                return
            
            train_ratio, val_ratio, test_ratio = splits
            total = sum(splits)
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total
            
            logger.info(f"Split ratios - Train: {train_ratio:.2f}, Val: {val_ratio:.2f}, Test: {test_ratio:.2f}")
            
            # Perform stratified split by event_type
            if 'synthetic' in self.annotations.columns:
                # For datasets with synthetic data, keep synthetic data in training set
                synthetic_data = self.annotations[self.annotations['synthetic'] == True]
                real_data = self.annotations[self.annotations['synthetic'] == False]
                
                # Split real data
                train_data, temp_data = train_test_split(
                    real_data, 
                    train_size=train_ratio/(val_ratio + test_ratio + train_ratio),
                    stratify=real_data['event_type'],
                    random_state=self.args.seed
                )
                
                # Further split temp_data into val and test
                val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
                val_data, test_data = train_test_split(
                    temp_data,
                    train_size=val_ratio_adjusted,
                    stratify=temp_data['event_type'],
                    random_state=self.args.seed
                )
                
                # Add synthetic data to training set
                train_data = pd.concat([train_data, synthetic_data], ignore_index=True)
                
            else:
                # Standard stratified split
                train_data, temp_data = train_test_split(
                    self.annotations, 
                    train_size=train_ratio,
                    stratify=self.annotations['event_type'],
                    random_state=self.args.seed
                )
                
                # Further split temp_data into val and test
                val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
                val_data, test_data = train_test_split(
                    temp_data,
                    train_size=val_ratio_adjusted,
                    stratify=temp_data['event_type'],
                    random_state=self.args.seed
                )
            
            # Save the splits
            train_data.to_csv(os.path.join(self.annotations_dir, "train_annotations.csv"), index=False)
            val_data.to_csv(os.path.join(self.annotations_dir, "val_annotations.csv"), index=False)
            test_data.to_csv(os.path.join(self.annotations_dir, "test_annotations.csv"), index=False)
            
            logger.info(f"Created splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
            
            # Create dataset files with split information
            self._create_dataset_files(train_data, val_data, test_data)
            
        except Exception as e:
            logger.error(f"Error creating splits: {str(e)}")
    
    def _create_dataset_files(self, train_data, val_data, test_data):
        """
        Create dataset files with split information.
        
        Args:
            train_data: Training data dataframe
            val_data: Validation data dataframe
            test_data: Test data dataframe
        """
        # Create a YAML file with dataset information
        dataset_info = {
            'name': 'UHCTD',
            'version': datetime.now().strftime('%Y%m%d'),
            'description': 'Self-Guiding Sentinels Dataset',
            'splits': {
                'train': {
                    'num_samples': len(train_data),
                    'by_class': train_data['event_type'].value_counts().to_dict()
                },
                'val': {
                    'num_samples': len(val_data),
                    'by_class': val_data['event_type'].value_counts().to_dict()
                },
                'test': {
                    'num_samples': len(test_data),
                    'by_class': test_data['event_type'].value_counts().to_dict()
                }
            },
            'classes': {
                'normal': 0,
                'environmental': 1,
                'maintenance': 2,
                'attack': 3
            }
        }
        
        # Save as JSON
        with open(os.path.join(self.output_dir, "dataset_info.json"), 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        logger.info("Created dataset information file")
    
    def _create_metadata(self):
        """Create metadata files."""
        logger.info("Creating metadata...")
        
        # Create class distribution visualization
        try:
            class_distribution = self.annotations['event_type'].value_counts()
            class_distribution.to_csv(os.path.join(self.metadata_dir, "class_distribution.csv"))
            logger.info("Created class distribution metadata")
            
            # Create video statistics
            video_stats = self.annotations.groupby('video_name').agg({
                'start_frame': 'min',
                'end_frame': 'max',
                'event_type': 'first'
            })
            video_stats['num_frames'] = video_stats['end_frame'] - video_stats['start_frame']
            video_stats.to_csv(os.path.join(self.metadata_dir, "video_statistics.csv"))
            logger.info("Created video statistics metadata")
            
            # Create dataset summary
            summary = {
                'num_videos': len(self.annotations['video_name'].unique()),
                'num_frames': self.annotations['end_frame'].sum() - self.annotations['start_frame'].sum(),
                'class_distribution': class_distribution.to_dict(),
                'synthetic_data': 'synthetic' in self.annotations.columns and any(self.annotations['synthetic']),
                'processed_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(os.path.join(self.metadata_dir, "dataset_summary.json"), 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info("Created dataset summary metadata")
            
        except Exception as e:
            logger.error(f"Error creating metadata: {str(e)}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process UHCTD dataset for Self-Guiding Sentinels")
    
    parser.add_argument("--uhctd-dir", type=str, required=True, help="Path to UHCTD dataset directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--synthetic-dir", type=str, default=None, help="Path to synthetic data directory")
    parser.add_argument("--splits-ratio", type=str, default="70:15:15", help="Train:Val:Test split ratio")
    parser.add_argument("--extract-frames", action="store_true", help="Extract frames from videos")
    parser.add_argument("--frame-interval", type=int, default=5, help="Interval between extracted frames")
    parser.add_argument("--create-metadata", action="store_true", help="Create dataset metadata")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    processor = DatasetProcessor(args)
    processor.process()
