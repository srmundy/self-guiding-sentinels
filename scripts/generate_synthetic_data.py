#!/usr/bin/env python
"""
Synthetic Data Generator for Self-Guiding Sentinels

This script generates synthetic data to enhance training for the Self-Guiding Sentinels
surveillance system. It creates realistic tampering scenarios using various techniques:
1. Augmentation of existing UHCTD dataset samples
2. Generation of attack patterns using diffusion models
3. Simulation of various tampering techniques (spray paint, lens covering, etc.)
4. Creation of realistic lighting and environmental variations

Usage:
    python generate_synthetic_data.py --output-dir data/synthetic --num-samples 1000
"""

import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps
import pandas as pd
import json
import time
import math
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Try importing the diffusion components, with graceful fallback
try:
    from diffusers import DDPMScheduler, UNet2DModel
    DIFFUSION_AVAILABLE = True
except ImportError:
    print("Warning: diffusers package not found. Will use basic augmentation techniques only.")
    print("To install: pip install diffusers")
    DIFFUSION_AVAILABLE = False


class SyntheticDataGenerator:
    """
    Generator for synthetic tampering data to enhance model training.
    """
    def __init__(self, args):
        """
        Initialize the synthetic data generator.
        
        Args:
            args: Command line arguments
        """
        self.args = args
        self.output_dir = args.output_dir
        self.num_samples = args.num_samples
        self.img_size = args.img_size
        self.seed = args.seed
        self.source_videos = []
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "videos"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "frames"), exist_ok=True)
        
        # Initialize diffusion model if available
        self.diffusion_model = None
        if DIFFUSION_AVAILABLE and args.use_diffusion:
            self._init_diffusion_model()
        
        # Load source data if specified
        if args.source_dir:
            self._load_source_data()
    
    def _init_diffusion_model(self):
        """Initialize the diffusion model for synthetic generation."""
        print("Initializing diffusion model...")
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize DDPM scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        # Initialize UNet model
        self.diffusion_model = UNet2DModel(
            sample_size=self.img_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        
        # Load pre-trained weights if available
        if self.args.diffusion_weights:
            print(f"Loading diffusion model weights from {self.args.diffusion_weights}")
            self.diffusion_model.load_state_dict(torch.load(self.args.diffusion_weights))
        
        self.diffusion_model = self.diffusion_model.to(device)
        self.diffusion_model.eval()
        print(f"Diffusion model initialized on {device}")
    
    def _load_source_data(self):
        """Load source data from the specified directory."""
        source_dir = self.args.source_dir
        print(f"Loading source data from {source_dir}")
        
        # Load video files
        video_dir = os.path.join(source_dir, "videos")
        if os.path.exists(video_dir):
            video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) 
                          if f.endswith(('.mp4', '.avi', '.mov'))]
            self.source_videos = video_files
            print(f"Found {len(self.source_videos)} source videos")
        
        # Load annotations if available
        annotation_file = os.path.join(source_dir, "annotations.csv")
        if os.path.exists(annotation_file):
            self.annotations = pd.read_csv(annotation_file)
            print(f"Loaded annotations with {len(self.annotations)} entries")
        else:
            self.annotations = None
            print("No annotation file found")
    
    def generate_data(self):
        """Main method to generate synthetic data."""
        print(f"Generating {self.num_samples} synthetic samples...")
        
        # Initialize annotation dataframe
        annotations = []
        
        # Generate samples
        for i in tqdm(range(self.num_samples)):
            # Generate a unique ID for this sample
            sample_id = f"synthetic_{i:06d}"
            
            # Determine the tampering type for this sample
            tampering_type = self._select_tampering_type()
            
            # Generate the synthetic video
            output_path, metadata = self._generate_sample(sample_id, tampering_type)
            
            # Record the annotation
            annotations.append({
                'video_name': os.path.basename(output_path),
                'event_type': tampering_type,
                'start_frame': metadata['start_frame'],
                'end_frame': metadata['end_frame'],
                'synthetic': True,
                'generation_method': metadata['method']
            })
        
        # Save annotations to CSV
        annotations_df = pd.DataFrame(annotations)
        annotations_path = os.path.join(self.output_dir, "synthetic_annotations.csv")
        annotations_df.to_csv(annotations_path, index=False)
        print(f"Saved annotations to {annotations_path}")
        
        # Generate visualization of the dataset
        self._visualize_dataset(annotations_df)
        
        return annotations_df
    
    def _select_tampering_type(self):
        """
        Select a tampering type for the synthetic sample.
        
        Returns:
            str: Tampering type ("normal", "environmental", "maintenance", or "attack")
        """
        if self.args.class_distribution:
            # Parse the class distribution from command line
            distribution = {}
            for item in self.args.class_distribution.split(','):
                class_name, prob = item.split(':')
                distribution[class_name] = float(prob)
            
            # Normalize probabilities if needed
            total = sum(distribution.values())
            if total != 1.0:
                for k in distribution:
                    distribution[k] /= total
            
            # Select according to distribution
            classes = list(distribution.keys())
            probs = [distribution[c] for c in classes]
            return np.random.choice(classes, p=probs)
        else:
            # Default distribution (focus on attack samples)
            return np.random.choice(
                ["normal", "environmental", "maintenance", "attack"], 
                p=[0.1, 0.2, 0.2, 0.5]
            )
    
    def _generate_sample(self, sample_id, tampering_type):
        """
        Generate a synthetic sample with the specified tampering type.
        
        Args:
            sample_id: Unique identifier for the sample
            tampering_type: Type of tampering to simulate
            
        Returns:
            tuple: (output_path, metadata_dict)
        """
        # Determine generation method based on tampering type and available resources
        if tampering_type == "attack" and DIFFUSION_AVAILABLE and self.args.use_diffusion:
            method = "diffusion"
        elif self.source_videos and random.random() < 0.7:
            method = "augmentation"
        else:
            method = "procedural"
        
        # Generate the sample using the selected method
        if method == "diffusion":
            output_path, metadata = self._generate_with_diffusion(sample_id, tampering_type)
        elif method == "augmentation":
            output_path, metadata = self._generate_with_augmentation(sample_id, tampering_type)
        else:
            output_path, metadata = self._generate_procedural(sample_id, tampering_type)
        
        metadata['method'] = method
        return output_path, metadata
    
    def _generate_with_diffusion(self, sample_id, tampering_type):
        """
        Generate a synthetic sample using the diffusion model.
        
        Args:
            sample_id: Unique identifier for the sample
            tampering_type: Type of tampering to simulate
            
        Returns:
            tuple: (output_path, metadata_dict)
        """
        # Prepare output paths
        video_path = os.path.join(self.output_dir, "videos", f"{sample_id}.mp4")
        frames_dir = os.path.join(self.output_dir, "frames", sample_id)
        os.makedirs(frames_dir, exist_ok=True)
        
        # Video parameters
        fps = 30
        duration = random.uniform(3.0, 10.0)  # seconds
        num_frames = int(fps * duration)
        
        # Define start and end of tampering event
        start_frame = int(num_frames * 0.2)  # Start at 20% of the video
        end_frame = int(num_frames * 0.8)    # End at 80% of the video
        
        # Create the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            video_path, fourcc, fps, (self.img_size, self.img_size))
        
        # Generate frames with diffusion model
        device = next(self.diffusion_model.parameters()).device
        
        # Initial noise
        noise = torch.randn(1, 3, self.img_size, self.img_size).to(device)
        
        # Conditioning on tampering type
        # We'll use different noise levels based on tampering type
        if tampering_type == "attack":
            noise_level = torch.tensor([800]).long().to(device)
        elif tampering_type == "maintenance":
            noise_level = torch.tensor([600]).long().to(device)
        elif tampering_type == "environmental":
            noise_level = torch.tensor([400]).long().to(device)
        else:  # normal
            noise_level = torch.tensor([200]).long().to(device)
        
        # Generate the base frame (starting point)
        with torch.no_grad():
            # Start from random noise
            image = noise
            
            # Gradually denoise
            for t in reversed(range(0, 1000, 50)):
                timestep = torch.tensor([t]).long().to(device)
                noise_pred = self.diffusion_model(image, timestep).sample
                image = self.scheduler.step(noise_pred, t, image).prev_sample
        
        # Get the base frame
        base_frame = (image[0].permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        
        # Now generate the sequence by adding progressive tampering
        for i in range(num_frames):
            frame = base_frame.copy()
            
            # Apply tampering effects based on frame position
            if start_frame <= i <= end_frame:
                # Calculate tampering intensity (ramps up then down)
                if i <= (start_frame + end_frame) // 2:
                    intensity = (i - start_frame) / ((end_frame - start_frame) / 2)
                else:
                    intensity = (end_frame - i) / ((end_frame - start_frame) / 2)
                
                # Apply tampering effect
                frame = self._apply_tampering_effect(frame, tampering_type, intensity)
            
            # Save the frame
            frame_path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # Add to video
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Release the video writer
        video_writer.release()
        
        # Return metadata
        metadata = {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'fps': fps,
            'duration': duration,
            'num_frames': num_frames
        }
        
        return video_path, metadata
    
    def _generate_with_augmentation(self, sample_id, tampering_type):
        """
        Generate a synthetic sample by augmenting a source video.
        
        Args:
            sample_id: Unique identifier for the sample
            tampering_type: Type of tampering to simulate
            
        Returns:
            tuple: (output_path, metadata_dict)
        """
        # Prepare output paths
        video_path = os.path.join(self.output_dir, "videos", f"{sample_id}.mp4")
        frames_dir = os.path.join(self.output_dir, "frames", sample_id)
        os.makedirs(frames_dir, exist_ok=True)
        
        # Select a random source video
        source_video = random.choice(self.source_videos)
        
        # Open the video
        cap = cv2.VideoCapture(source_video)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            video_path, fourcc, fps, (self.img_size, self.img_size))
        
        # Define start and end of tampering event (random but within video length)
        start_frame = random.randint(int(frame_count * 0.1), int(frame_count * 0.4))
        end_frame = random.randint(int(frame_count * 0.6), int(frame_count * 0.9))
        
        # Process each frame
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            
            # Apply tampering effects based on frame position
            if start_frame <= frame_idx <= end_frame and tampering_type != "normal":
                # Calculate tampering intensity (ramps up then down)
                if frame_idx <= (start_frame + end_frame) // 2:
                    intensity = (frame_idx - start_frame) / ((end_frame - start_frame) / 2)
                else:
                    intensity = (end_frame - frame_idx) / ((end_frame - start_frame) / 2)
                
                # Apply tampering effect
                frame = self._apply_tampering_effect(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                    tampering_type, 
                    intensity
                )
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Save the frame
            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            
            # Add to video
            video_writer.write(frame)
            
            frame_idx += 1
        
        # Release resources
        cap.release()
        video_writer.release()
        
        # Return metadata
        metadata = {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'fps': fps,
            'source_video': source_video,
            'num_frames': frame_idx
        }
        
        return video_path, metadata
    
    def _generate_procedural(self, sample_id, tampering_type):
        """
        Generate a synthetic sample procedurally without source videos.
        
        Args:
            sample_id: Unique identifier for the sample
            tampering_type: Type of tampering to simulate
            
        Returns:
            tuple: (output_path, metadata_dict)
        """
        # Prepare output paths
        video_path = os.path.join(self.output_dir, "videos", f"{sample_id}.mp4")
        frames_dir = os.path.join(self.output_dir, "frames", sample_id)
        os.makedirs(frames_dir, exist_ok=True)
        
        # Video parameters
        fps = 30
        duration = random.uniform(3.0, 8.0)  # seconds
        num_frames = int(fps * duration)
        
        # Define start and end of tampering event
        start_frame = int(num_frames * 0.2)  # Start at 20% of the video
        end_frame = int(num_frames * 0.8)    # End at 80% of the video
        
        # Create the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            video_path, fourcc, fps, (self.img_size, self.img_size))
        
        # Generate a base scene
        base_frame = self._generate_base_scene()
        
        # Add surveillance elements to make it look like a camera view
        base_frame = self._add_surveillance_elements(base_frame)
        
        # Generate frames with procedural tampering
        for i in range(num_frames):
            frame = base_frame.copy()
            
            # Add slight movement/jitter to simulate a real camera
            frame = self._add_camera_movement(frame, i)
            
            # Apply tampering effects based on frame position
            if start_frame <= i <= end_frame and tampering_type != "normal":
                # Calculate tampering intensity (ramps up then down)
                if i <= (start_frame + end_frame) // 2:
                    intensity = (i - start_frame) / ((end_frame - start_frame) / 2)
                else:
                    intensity = (end_frame - i) / ((end_frame - start_frame) / 2)
                
                # Apply tampering effect
                frame = self._apply_tampering_effect(frame, tampering_type, intensity)
            
            # Convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Save the frame
            frame_path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(frame_path, frame_bgr)
            
            # Add to video
            video_writer.write(frame_bgr)
        
        # Release the video writer
        video_writer.release()
        
        # Return metadata
        metadata = {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'fps': fps,
            'duration': duration,
            'num_frames': num_frames
        }
        
        return video_path, metadata
    
    def _generate_base_scene(self):
        """
        Generate a base scene for procedural generation.
        
        Returns:
            numpy.ndarray: Base scene image
        """
        # Create a blank image
        img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 255
        
        # Randomly choose a scene type
        scene_type = random.choice(['indoor', 'outdoor', 'hallway', 'parking'])
        
        if scene_type == 'indoor':
            # Generate an indoor scene (room with walls, floor, ceiling)
            # Floor
            cv2.rectangle(img, (0, self.img_size//2), (self.img_size, self.img_size), 
                         (200, 200, 200), -1)
            # Back wall
            cv2.rectangle(img, (0, 0), (self.img_size, self.img_size//2), 
                         (220, 220, 220), -1)
            # Random objects
            for _ in range(random.randint(2, 5)):
                x = random.randint(50, self.img_size-100)
                y = random.randint(self.img_size//2, self.img_size-50)
                w = random.randint(30, 100)
                h = random.randint(30, 100)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
        
        elif scene_type == 'outdoor':
            # Sky
            cv2.rectangle(img, (0, 0), (self.img_size, self.img_size//2), 
                         (135, 206, 235), -1)
            # Ground
            cv2.rectangle(img, (0, self.img_size//2), (self.img_size, self.img_size), 
                         (34, 139, 34), -1)
            # Random buildings
            for _ in range(random.randint(1, 3)):
                x = random.randint(50, self.img_size-150)
                y = random.randint(100, self.img_size//2-20)
                w = random.randint(100, 200)
                h = random.randint(150, self.img_size//2-y)
                color = (random.randint(100, 180), random.randint(100, 180), random.randint(100, 180))
                cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
        
        elif scene_type == 'hallway':
            # Walls
            cv2.rectangle(img, (0, 0), (self.img_size, self.img_size), 
                         (220, 220, 220), -1)
            # Floor
            pts = np.array([[0, self.img_size//2], [self.img_size, self.img_size//2], 
                          [self.img_size*3//4, self.img_size], [self.img_size//4, self.img_size]])
            cv2.fillPoly(img, [pts], (180, 180, 180))
            # Ceiling
            pts = np.array([[0, self.img_size//2], [self.img_size, self.img_size//2], 
                          [self.img_size*3//4, 0], [self.img_size//4, 0]])
            cv2.fillPoly(img, [pts], (240, 240, 240))
            # Doors
            for _ in range(random.randint(1, 3)):
                x = random.randint(50, self.img_size-100)
                y = random.randint(self.img_size//4, self.img_size//2-20)
                w = random.randint(40, 80)
                h = random.randint(80, 150)
                color = (120, 80, 40)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
        
        elif scene_type == 'parking':
            # Pavement
            cv2.rectangle(img, (0, 0), (self.img_size, self.img_size), 
                         (100, 100, 100), -1)
            # Parking lines
            line_spacing = random.randint(60, 100)
            for i in range(0, self.img_size, line_spacing):
                cv2.line(img, (i, 0), (i, self.img_size), (250, 250, 250), 2)
            # Random cars
            for _ in range(random.randint(0, 4)):
                x = random.randint(20, self.img_size-80)
                y = random.randint(20, self.img_size-40)
                w = random.randint(60, 80)
                h = random.randint(30, 40)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
        
        # Convert to PIL for further processing
        pil_img = Image.fromarray(img)
        
        # Add some texture and noise
        pil_img = self._add_texture(pil_img)
        
        # Convert back to numpy array
        return np.array(pil_img)
    
    def _add_texture(self, pil_img):
        """
        Add texture and noise to an image to make it more realistic.
        
        Args:
            pil_img: PIL Image to process
            
        Returns:
            PIL.Image: Processed image
        """
        # Add noise
        noise = Image.new('L', pil_img.size)
        draw = ImageDraw.Draw(noise)
        
        for x in range(0, pil_img.width, 2):
            for y in range(0, pil_img.height, 2):
                draw.point((x, y), fill=random.randint(0, 50))
        
        # Convert to RGB if it's not already
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        # Create the noise overlay
        noise_overlay = Image.new('RGB', pil_img.size)
        noise_overlay.paste(noise, (0, 0))
        
        # Blend with original
        return Image.blend(pil_img, noise_overlay, 0.1)
    
    def _add_surveillance_elements(self, img):
        """
        Add surveillance camera elements to make the image look like a camera feed.
        
        Args:
            img: Base image (numpy array)
            
        Returns:
            numpy.ndarray: Image with surveillance elements
        """
        # Convert to PIL for easier text drawing
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        
        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        draw.text((10, 10), timestamp, fill=(255, 255, 255))
        
        # Add camera ID
        camera_id = f"CAM-{random.randint(1, 99):02d}"
        draw.text((self.img_size - 80, 10), camera_id, fill=(255, 255, 255))
        
        # Optionally add recording indicator
        if random.random() < 0.7:
            draw.ellipse((10, 30, 20, 40), fill=(255, 0, 0))
            draw.text((25, 30), "REC", fill=(255, 255, 255))
        
        # Convert back to numpy array
        return np.array(pil_img)
    
    def _add_camera_movement(self, img, frame_num):
        """
        Add slight camera movement to simulate a real camera.
        
        Args:
            img: Image to process
            frame_num: Current frame number
            
        Returns:
            numpy.ndarray: Image with camera movement
        """
        # Convert to PIL
        pil_img = Image.fromarray(img)
        
        # Apply very slight random shift
        shift_x = math.sin(frame_num * 0.1) * 2
        shift_y = math.cos(frame_num * 0.1) * 2
        
        # Use affine transform for the shift
        pil_img = pil_img.transform(
            pil_img.size, 
            Image.AFFINE, 
            (1, 0, shift_x, 0, 1, shift_y),
            resample=Image.BICUBIC
        )
        
        return np.array(pil_img)
    
    def _apply_tampering_effect(self, img, tampering_type, intensity):
        """
        Apply a tampering effect to the image based on the tampering type.
        
        Args:
            img: Image to process (numpy array)
            tampering_type: Type of tampering to simulate
            intensity: Intensity of the effect (0.0 to 1.0)
            
        Returns:
            numpy.ndarray: Processed image
        """
        # Convert to PIL for easier processing
        pil_img = Image.fromarray(img.astype(np.uint8))
        
        if tampering_type == "attack":
            # Choose a random attack method
            attack_method = random.choice([
                "spray_paint", "covering", "camera_rotation", "smashing", "laser_pointing"
            ])
            
            if attack_method == "spray_paint":
                # Simulate spray paint by adding colored splotches
                draw = ImageDraw.Draw(pil_img)
                spray_color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )
                
                # Create multiple spray spots with varying opacity
                num_spots = int(30 * intensity)
                for _ in range(num_spots):
                    x = random.randint(0, self.img_size)
                    y = random.randint(0, self.img_size)
                    radius = random.randint(5, 20)
                    draw.ellipse((x-radius, y-radius, x+radius, y+radius), 
                               fill=(*spray_color, int(200 * intensity)))
            
            elif attack_method == "covering":
                # Simulate covering the camera lens
                overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, int(200 * intensity)))
                pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay)
            
            elif attack_method == "camera_rotation":
                # Simulate camera being physically rotated
                angle = 45 * intensity
                pil_img = pil_img.rotate(angle, resample=Image.BICUBIC, expand=False)
            
            elif attack_method == "smashing":
                # Simulate camera being smashed (cracks)
                
