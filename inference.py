#!/usr/bin/env python
"""
Real-time Inference Script for Self-Guiding Sentinels

This script performs real-time tampering detection on video streams (camera or file)
and implements the graduated threat level response system.

Usage:
    python inference.py --model-path models/best_model.pth --input-source 0
    python inference.py --model-path models/best_model.pth --input-source video.mp4 --output-video output.mp4
"""

import os
import sys
import time
import argparse
import json
import yaml
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
from collections import deque
import logging
from datetime import datetime
import threading
import signal
import pygame  # For audio playback
from threading import Thread
from queue import Queue, Empty

# Import the model architecture
from model import SelfGuidingSentinels
from utils.data_loader import get_transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('inference.log')
    ]
)

logger = logging.getLogger(__name__)

# Class names
CLASS_NAMES = {
    0: "Normal Operation",
    1: "Environmental Factors",
    2: "Legitimate Maintenance",
    3: "Physical Attack"
}

# Class colors (BGR for OpenCV)
CLASS_COLORS = {
    0: (0, 255, 0),      # Green
    1: (0, 255, 255),    # Yellow
    2: (0, 165, 255),    # Orange
    3: (0, 0, 255)       # Red
}

# Threat levels
THREAT_LEVELS = {
    0: {
        "name": "Normal",
        "confidence_range": [0.0, 0.5],
        "color": (0, 255, 0),
        "audio": None
    },
    1: {
        "name": "Monitoring",
        "confidence_range": [0.5, 0.65],
        "color": (0, 255, 255),
        "audio": "sounds/monitoring.wav"
    },
    2: {
        "name": "Warning",
        "confidence_range": [0.65, 0.8],
        "color": (0, 165, 255),
        "audio": "sounds/warning.wav"
    },
    3: {
        "name": "Deterrence",
        "confidence_range": [0.8, 0.9],
        "color": (0, 0, 255),
        "audio": "sounds/deterrence.wav"
    },
    4: {
        "name": "Critical",
        "confidence_range": [0.9, 1.0],
        "color": (0, 0, 128),
        "audio": "sounds/pain.wav"
    }
}


class AudioPlayer:
    """
    Class for handling audio responses based on threat levels
    """
    def __init__(self, audio_dir="sounds"):
        """
        Initialize the audio player
        
        Args:
            audio_dir: Directory containing audio files
        """
        self.audio_dir = audio_dir
        self.sound_files = {}
        self.current_level = 0
        self.last_played = 0
        self.cooldown = 3.0  # Seconds between audio responses
        
        # Initialize pygame for audio
        try:
            pygame.mixer.init()
            logger.info("Audio system initialized")
            
            # Load audio files
            self._load_audio_files()
        except Exception as e:
            logger.error(f"Failed to initialize audio system: {str(e)}")
            self.audio_enabled = False
        else:
            self.audio_enabled = True
    
    def _load_audio_files(self):
        """Load audio files for each threat level"""
        for level, data in THREAT_LEVELS.items():
            audio_file = data.get("audio")
            if audio_file:
                file_path = os.path.join(self.audio_dir, os.path.basename(audio_file))
                if os.path.exists(file_path):
                    try:
                        self.sound_files[level] = pygame.mixer.Sound(file_path)
                        logger.info(f"Loaded audio file for threat level {level}: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to load audio file {file_path}: {str(e)}")
    
    def play_for_threat_level(self, level):
        """
        Play audio response for the given threat level
        
        Args:
            level: Threat level (0-4)
        """
        if not self.audio_enabled:
            return
        
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_played < self.cooldown:
            return
        
        # Only play if threat level increased or it's a high-level threat
        if level <= self.current_level and level < 3:
            return
        
        # Play sound
        sound = self.sound_files.get(level)
        if sound:
            try:
                pygame.mixer.stop()  # Stop any currently playing sounds
                sound.play()
                self.last_played = current_time
                self.current_level = level
                logger.info(f"Playing audio for threat level {level}")
            except Exception as e:
                logger.error(f"Failed to play audio: {str(e)}")
    
    def cleanup(self):
        """Clean up resources"""
        if self.audio_enabled:
            pygame.mixer.quit()


class SentinelInference:
    """
    Class for real-time inference using the Self-Guiding Sentinels model
    """
    def __init__(self, args):
        """
        Initialize the inference system
        
        Args:
            args: Command line arguments
        """
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.running = False
        self.frames_processed = 0
        self.start_time = 0
        
        # Initialize model
        self._load_model()
        
        # Initialize video source
        self._setup_video_source()
        
        # Initialize video writer if output specified
        self._setup_video_writer()
        
        # Initialize audio player if enabled
        if args.audio_response:
            self.audio_player = AudioPlayer(args.audio_dir)
        else:
            self.audio_player = None
        
        # Initialize detection history for temporal smoothing
        self.detection_history = deque(maxlen=args.history_size)
        
        # Setup result logging
        if args.log_detections:
            self.detection_log = []
        
        # For threat level system
        self.current_threat_level = 0
        self.threat_level_history = deque(maxlen=args.history_size)
        self.threat_start_time = None
    
    def _load_model(self):
        """Load the Self-Guiding Sentinels model"""
        logger.info(f"Loading model from {self.args.model_path}")
        try:
            # Initialize model architecture
            self.model = SelfGuidingSentinels(
                img_size=self.args.img_size,
                patch_size=self.args.patch_size,
                in_channels=3,
                embed_dim=self.args.embed_dim,
                vit_depth=self.args.vit_depth,
                n_heads=self.args.n_heads,
                mlp_ratio=self.args.mlp_ratio,
                diffusion_hidden_dim=self.args.diffusion_hidden_dim,
                n_classes=len(CLASS_NAMES),
                dropout=0.0  # No dropout during inference
            )
            
            # Load weights
            checkpoint = torch.load(self.args.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            
            # Move model to device and set to evaluation mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Initialize the preprocessing transform
            self.transform = get_transforms(is_training=False)
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            sys.exit(1)
    
    def _setup_video_source(self):
        """Set up the video source (camera or file)"""
        source = self.args.input_source
        try:
            # Handle different input source types
            if source.isdigit():
                # Camera index
                self.cap = cv2.VideoCapture(int(source))
                logger.info(f"Opened camera with index {source}")
            elif os.path.exists(source):
                # Video file
                self.cap = cv2.VideoCapture(source)
                logger.info(f"Opened video file: {source}")
            elif source.startswith(('rtsp://', 'http://', 'https://')):
                # Streaming URL
                self.cap = cv2.VideoCapture(source)
                logger.info(f"Opened video stream: {source}")
            else:
                logger.error(f"Invalid input source: {source}")
                sys.exit(1)
            
            # Check if source was opened successfully
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {source}")
                sys.exit(1)
            
            # Get video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0 or self.fps == float('inf'):
                self.fps = 30  # Default FPS if not available
            
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Video source properties: {self.frame_width}x{self.frame_height} @ {self.fps} FPS")
        except Exception as e:
            logger.error(f"Error setting up video source: {str(e)}")
            sys.exit(1)
    
    def _setup_video_writer(self):
        """Set up video writer for output if specified"""
        if self.args.output_video:
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    self.args.output_video,
                    fourcc,
                    self.fps,
                    (self.frame_width, self.frame_height)
                )
                logger.info(f"Output video will be saved to: {self.args.output_video}")
            except Exception as e:
                logger.error(f"Failed to create video writer: {str(e)}")
                self.video_writer = None
        else:
            self.video_writer = None
    
    def run(self):
        """Run the inference loop"""
        logger.info("Starting inference...")
        self.running = True
        self.start_time = time.time()
        self.frames_processed = 0
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            while self.running and self.cap.isOpened():
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.info("End of video stream reached")
                    break
                
                # Process frame
                result_frame, detection_result = self.process_frame(frame)
                
                # Display frame if not in headless mode
                if not self.args.headless:
                    cv2.imshow("Self-Guiding Sentinels", result_frame)
                    
                    # Check for key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("User requested exit")
                        break
                
                # Write frame to output video if enabled
                if self.video_writer is not None:
                    self.video_writer.write(result_frame)
                
                # Log detections if enabled
                if self.args.log_detections and detection_result:
                    self.detection_log.append(detection_result)
                
                # Update statistics
                self.frames_processed += 1
                
                # Limit frame rate if specified
                if self.args.max_fps > 0:
                    elapsed = time.time() - self.start_time
                    expected_elapsed = self.frames_processed / self.args.max_fps
                    if elapsed < expected_elapsed:
                        time.sleep(expected_elapsed - elapsed)
            
            # Calculate and log performance statistics
            self._log_performance_stats()
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            self.cleanup()
    
    def process_frame(self, frame):
        """
        Process a single frame for tampering detection
        
        Args:
            frame: OpenCV frame (BGR format)
        
        Returns:
            tuple: (processed_frame, detection_result)
        """
        # Start timing
        inference_start = time.time()
        
        # Make a copy of the frame for visualization
        result_frame = frame.copy()
        
        # Prepare the frame for the model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Resize and apply transforms
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            
            # Get class prediction and confidence
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()
            confidence = probs[0, predicted_class].item()
        
        # Measure inference time
        inference_time = (time.time() - inference_start) * 1000  # ms
        
        # Create detection result
        detection_result = {
            "timestamp": datetime.now().isoformat(),
            "frame_number": self.frames_processed,
            "predicted_class": predicted_class,
            "class_name": CLASS_NAMES[predicted_class],
            "confidence": confidence,
            "inference_time_ms": inference_time,
            "probabilities": {CLASS_NAMES[i]: probs[0, i].item() for i in range(len(CLASS_NAMES))}
        }
        
        # Add to detection history for temporal smoothing
        self.detection_history.append((predicted_class, confidence))
        
        # Apply temporal smoothing if enough history
        smoothed_class, smoothed_confidence = self._apply_temporal_smoothing()
        detection_result["smoothed_class"] = smoothed_class
        detection_result["smoothed_confidence"] = smoothed_confidence
        detection_result["smoothed_class_name"] = CLASS_NAMES[smoothed_class]
        
        # Determine threat level
        threat_level = self._determine_threat_level(smoothed_class, smoothed_confidence)
        detection_result["threat_level"] = threat_level
        detection_result["threat_level_name"] = THREAT_LEVELS[threat_level]["name"]
        
        # Trigger audio response if enabled
        if self.audio_player and self.args.audio_response:
            self.audio_player.play_for_threat_level(threat_level)
        
        # Visualize results on the frame
        result_frame = self._visualize_detection(result_frame, detection_result)
        
        return result_frame, detection_result
    
    def _apply_temporal_smoothing(self):
        """
        Apply temporal smoothing to detection results
        
        Returns:
            tuple: (smoothed_class, smoothed_confidence)
        """
        if len(self.detection_history) < 3:
            # Not enough history, return last prediction
            return self.detection_history[-1]
        
        # Count class frequencies
        class_counts = {}
        total_conf_by_class = {}
        
        for cls, conf in self.detection_history:
            class_counts[cls] = class_counts.get(cls, 0) + 1
            total_conf_by_class[cls] = total_conf_by_class.get(cls, 0) + conf
        
        # Find the most frequent class
        max_count = 0
        max_class = 0
        
        for cls, count in class_counts.items():
            if count > max_count:
                max_count = count
                max_class = cls
        
        # Calculate average confidence for the most frequent class
        avg_confidence = total_conf_by_class[max_class] / class_counts[max_class]
        
        return max_class, avg_confidence
    
    def _determine_threat_level(self, predicted_class, confidence):
        """
        Determine the current threat level based on class and confidence
        
        Args:
            predicted_class: Class index
            confidence: Confidence score
            
        Returns:
            int: Threat level (0-4)
        """
        # Only consider attack class for threat levels 1-4
        if predicted_class == 3:  # Physical Attack class
            # Find appropriate threat level based on confidence
            for level, data in THREAT_LEVELS.items():
                min_conf, max_conf = data["confidence_range"]
                if min_conf <= confidence < max_conf:
                    # We found the threat level
                    threat_level = level
                    break
            else:
                # Default to highest if nothing matched
                threat_level = 4
        elif predicted_class == 2:  # Maintenance class
            # For maintenance, cap threat level at 2 (Warning)
            confidence_adjusted = min(confidence, 0.79)
            for level, data in THREAT_LEVELS.items():
                if level <= 2:  # Only consider levels 0-2 for maintenance
                    min_conf, max_conf = data["confidence_range"]
                    if min_conf <= confidence_adjusted < max_conf:
                        threat_level = level
                        break
            else:
                threat_level = 2
        elif predicted_class == 1:  # Environmental factors
            # For environmental, cap threat level at 1 (Monitoring)
            confidence_adjusted = min(confidence, 0.64)
            for level, data in THREAT_LEVELS.items():
                if level <= 1:  # Only consider levels 0-1 for environmental
                    min_conf, max_conf = data["confidence_range"]
                    if min_conf <= confidence_adjusted < max_conf:
                        threat_level = level
                        break
            else:
                threat_level = 1
        else:  # Normal operation
            threat_level = 0
        
        # Add to threat history
        self.threat_level_history.append(threat_level)
        
        # Apply temporal hysteresis to avoid rapid oscillation
        # Only increase threat level immediately, decrease gradually
        if len(self.threat_level_history) >= 3:
            if threat_level > self.current_threat_level:
                # Increase immediately if confirmed by multiple frames
                high_level_count = sum(1 for t in self.threat_level_history[-3:] if t >= threat_level)
                if high_level_count >= 2:
                    self.current_threat_level = threat_level
                    # Record start time for new threat level
                    if self.threat_start_time is None:
                        self.threat_start_time = time.time()
            elif threat_level < self.current_threat_level:
                # Decrease only if consistently lower for multiple frames
                low_level_count = sum(1 for t in self.threat_level_history[-5:] if t <= threat_level)
                if low_level_count >= 4:
                    self.current_threat_level = threat_level
                    # Reset start time
                    self.threat_start_time = None
        else:
            # Not enough history, set directly
            self.current_threat_level = threat_level
        
        return self.current_threat_level
    
    def _visualize_detection(self, frame, detection_result):
        """
        Visualize detection results on the frame
        
        Args:
            frame: Original frame
            detection_result: Detection results dictionary
            
        Returns:
            numpy.ndarray: Frame with visualization
        """
        # Get relevant data
        smoothed_class = detection_result["smoothed_class"]
        smoothed_confidence = detection_result["smoothed_confidence"]
        class_name = detection_result["smoothed_class_name"]
        threat_level = detection_result["threat_level"]
        threat_name = detection_result["threat_level_name"]
        
        # Get colors
        class_color = CLASS_COLORS[smoothed_class]
        threat_color = THREAT_LEVELS[threat_level]["color"]
        
        # Draw semi-transparent overlay based on threat level
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), threat_color, -1)
        alpha = min(0.3, threat_level * 0.075)  # Opacity increases with threat level
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw class label and confidence
        label = f"{class_name}: {smoothed_confidence:.2f}"
        cv2.putText(
            frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, class_color, 2, cv2.LINE_AA
        )
        
        # Draw threat level
        threat_label = f"Threat Level {threat_level}: {threat_name}"
        cv2.putText(
            frame, threat_label, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, threat_color, 2, cv2.LINE_AA
        )
        
        # Draw inference time
        inference_time = detection_result["inference_time_ms"]
        fps_text = f"Inference: {inference_time:.1f}ms ({1000/inference_time:.1f} FPS)"
        cv2.putText(
            frame, fps_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, (255, 255, 255), 2, cv2.LINE_AA
        )
        
        # Draw timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        cv2.putText(
            frame, timestamp, (frame.shape[1] - 230, frame.shape[0] - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
        )
        
        # Draw threat indicator
        indicator_width = int(frame.shape[1] * 0.8)
        indicator_height = 20
        indicator_x = int((frame.shape[1] - indicator_width) / 2)
        indicator_y = frame.shape[0] - 40
        
        # Draw background
        cv2.rectangle(
            frame, 
            (indicator_x, indicator_y), 
            (indicator_x + indicator_width, indicator_y + indicator_height), 
            (50, 50, 50), 
            -1
        )
        
        # Draw threat level segments
        segment_width = indicator_width / 5
        for i in range(5):
            segment_color = THREAT_LEVELS[i]["color"]
            start_x = indicator_x + int(i * segment_width)
            end_x = indicator_x + int((i + 1) * segment_width)
            
            # Fill up to current threat level
            if i <= threat_level:
                cv2.rectangle(
                    frame, 
                    (start_x, indicator_y), 
                    (end_x, indicator_y + indicator_height), 
                    segment_color, 
                    -1
                )
            
            # Draw segment borders
            cv2.line(
                frame, 
                (start_x, indicator_y), 
                (start_x, indicator_y + indicator_height), 
                (200, 200, 200), 
                1
            )
        
        # Draw outer border
        cv2.rectangle(
            frame, 
            (indicator_x, indicator_y), 
            (indicator_x + indicator_width, indicator_y + indicator_height), 
            (200, 200, 200), 
            1
        )
        
        return frame
    
    def _log_performance_stats(self):
        """Log performance statistics"""
        total_time = time.time() - self.start_time
        if total_time > 0 and self.frames_processed > 0:
            avg_fps = self.frames_processed / total_time
            logger.info(f"Processed {self.frames_processed} frames in {total_time:.2f} seconds")
            logger.info(f"Average FPS: {avg_fps:.2f}")
    
    def _signal_handler(self, sig, frame):
        """Handle signals for graceful shutdown"""
        logger.info(f"Received signal {sig}, shutting down...")
        self.running = False
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources...")
        
        # Release video capture
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        
        # Release video writer
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        
        # Clean up audio player
        if self.audio_player:
            self.audio_player.cleanup()
        
        # Save detection log if enabled
        if self.args.log_detections and self.detection_log:
            log_path = self.args.log_file or f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            try:
                with open(log_path, 'w') as f:
                    json.dump(self.detection_log, f, indent=2)
                logger.info(f"Detection log saved to {log_path}")
            except Exception as e:
                logger.error(f"Failed to save detection log: {str(e)}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Self-Guiding Sentinels Inference")
    
    # Model parameters
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size for ViT")
    parser.add_argument("--embed-dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--vit-depth", type=int, default=12, help="Depth of ViT")
    parser.add_argument("--n-heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--mlp-ratio", type=float, default=4.0, help="MLP ratio")
    parser.add_argument("--diffusion-hidden-dim", type=int, default=512, help="Diffusion hidden dim")
    
    # Input/output parameters
    parser.add_argument("--input-source", type=str, required=True, 
                        help="Input source (camera index, video file, or RTSP URL)")
    parser.add_argument("--output-video", type=str, default=None, help="Path to save output video")
    
    # Inference parameters
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda, cpu)")
    parser.add_argument("--max-fps", type=float, default=0, help="Maximum FPS (0 for no limit)")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--history-size", type=int, default=10, help="History size for temporal smoothing")
    
    # Response system parameters
    parser.add_argument("--threat-level-system", action="store_true", help="Enable threat level system")
    parser.add_argument("--audio-response", action="store_true", help="Enable audio responses")
    parser.add_argument("--audio-dir", type=str, default="sounds", help="Directory with audio files")
    
    # Logging parameters
    parser.add_argument("--log-detections", action="store_true", help="Log detection results")
    parser.add_argument("--log-file", type=str, default=None, help="Path to save detection log")
    
    # Parse arguments
    args = parser.parse_args()
    return args


def main():
    """Main function"""
    args = parse_args()
    
    # Create and run inference system
    inference = SentinelInference(args)
    try:
        inference.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    finally:
        inference.cleanup()


if __name__ == "__main__":
    main()
