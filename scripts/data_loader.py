import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import random
from PIL import Image, ImageFilter, ImageEnhance


class UHCTDDataset(Dataset):
    """
    Dataset class for UHCTD (University of Houston Camera Tampering Detection) dataset.
    """
    def __init__(self, root_dir, annotation_file, transform=None, frame_interval=5, clip_length=16):
        """
        Initialize the dataset.
        
        Args:
            root_dir (str): Directory with all the video files.
            annotation_file (str): Path to the annotation CSV file.
            transform (callable, optional): Optional transform to be applied on frames.
            frame_interval (int): Number of frames to skip between sampled frames.
            clip_length (int): Number of frames to include in each clip.
        """
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform
        self.frame_interval = frame_interval
        self.clip_length = clip_length
        
        # Map labels to numerical indices
        self.label_map = {
            'normal': 0,
            'environmental': 1,
            'maintenance': 2,
            'attack': 3
        }
        
        # Process annotations to get video clips
        self.clips = self._process_annotations()
    
    def _process_annotations(self):
        """
        Process the annotations to create a list of clips with their labels.
        
        Returns:
            List of tuples (video_path, start_frame, label)
        """
        clips = []
        
        for idx, row in self.annotations.iterrows():
            video_path = os.path.join(self.root_dir, row['video_name'])
            label = self.label_map[row['event_type']]
            
            # Get start and end frames
            start_frame = row['start_frame']
            end_frame = row['end_frame']
            
            # Create clips from the video
            for clip_start in range(start_frame, end_frame - self.clip_length * self.frame_interval, 
                                   self.clip_length * self.frame_interval // 2):
                clips.append((video_path, clip_start, label))
        
        return clips
    
    def _load_frames(self, video_path, start_frame):
        """
        Load a sequence of frames from a video.
        
        Args:
            video_path (str): Path to the video file.
            start_frame (int): Starting frame index.
            
        Returns:
            List of frames as numpy arrays.
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        # Set the starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read frames
        for i in range(self.clip_length):
            for _ in range(self.frame_interval):
                ret, frame = cap.read()
                if not ret:
                    # If we can't read more frames, duplicate the last one
                    if frames:
                        frames.append(frames[-1])
                    else:
                        # If no frames were read, reset and try again
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if ret:
                            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        # Ensure we have exactly clip_length frames
        if len(frames) < self.clip_length:
            # Duplicate the last frame if needed
            frames.extend([frames[-1]] * (self.clip_length - len(frames)))
        
        return frames[:self.clip_length]
    
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        """
        Get a clip of frames with its label.
        
        Args:
            idx (int): Index
            
        Returns:
            frames (torch.Tensor): Tensor of shape (C, T, H, W)
            label (int): Class label
        """
        video_path, start_frame, label = self.clips[idx]
        
        # Load frames
        frames = self._load_frames(video_path, start_frame)
        
        # Apply transforms
        if self.transform:
            transformed_frames = []
            for frame in frames:
                # Convert numpy array to PIL Image
                pil_frame = Image.fromarray(frame)
                # Apply transform
                transformed_frame = self.transform(pil_frame)
                transformed_frames.append(transformed_frame)
            
            # Stack frames along a new dimension
            frames_tensor = torch.stack(transformed_frames)
            
            # Rearrange from (T, C, H, W) to (C, T, H, W)
            frames_tensor = frames_tensor.permute(1, 0, 2, 3)
        else:
            # Convert to tensors without transforms
            frames_tensor = torch.stack([torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 for frame in frames])
            frames_tensor = frames_tensor.permute(1, 0, 2, 3)
        
        return frames_tensor, label


class SyntheticDataAugmentation:
    """
    Class for generating synthetic tampering data for augmentation.
    """
    def __init__(self, probability=0.5, intensity_range=(0.2, 0.8)):
        """
        Initialize the augmentation.
        
        Args:
            probability (float): Probability of applying each augmentation.
            intensity_range (tuple): Range of intensity values for augmentations.
        """
        self.probability = probability
        self.min_intensity, self.max_intensity = intensity_range
    
    def _random_intensity(self):
        """Get a random intensity value for augmentations."""
        return random.uniform(self.min_intensity, self.max_intensity)
    
    def simulate_tampering(self, image):
        """
        Apply random tampering effects to simulate physical attacks.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            PIL.Image: Augmented image
        """
        augmented = image
        
        # 1. Camera covering (darkening parts of the image)
        if random.random() < self.probability:
            # Create a black overlay for a portion of the image
            width, height = augmented.size
            overlay = Image.new('RGB', (width, height), (0, 0, 0))
            
            # Determine the region to be covered
            cover_width = int(width * self._random_intensity())
            cover_height = int(height * self._random_intensity())
            x_pos = random.randint(0, width - cover_width)
            y_pos = random.randint(0, height - cover_height)
            
            # Apply the overlay
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([x_pos, y_pos, x_pos + cover_width, y_pos + cover_height], fill=255)
            
            augmented = Image.composite(overlay, augmented, mask)
        
        # 2. Camera rotation (simulate camera being moved)
        if random.random() < self.probability:
            angle = random.uniform(-30, 30) * self._random_intensity()
            augmented = augmented.rotate(angle, resample=Image.BILINEAR)
        
        # 3. Lens smudging (simulate fingers on lens)
        if random.random() < self.probability:
            # Apply blur to simulate smudging
            blur_radius = 5 * self._random_intensity()
            augmented = augmented.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # 4. Spray painting (color overlay)
        if random.random() < self.probability:
            # Create a color overlay
            width, height = augmented.size
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            overlay = Image.new('RGB', (width, height), color)
            
            # Apply with random transparency
            alpha = self._random_intensity() * 0.7  # Reduce maximum opacity
            augmented = Image.blend(augmented, overlay, alpha)
        
        # 5. Camera jolting (motion blur)
        if random.random() < self.probability:
            # Apply motion blur
            kernel_size = int(10 * self._random_intensity()) * 2 + 1  # Ensure odd kernel size
            kernel = np.zeros((kernel_size, kernel_size))
            
            # Random direction of motion blur
            if random.random() < 0.5:
                # Horizontal motion blur
                kernel[kernel_size // 2, :] = 1.0
            else:
                # Vertical motion blur
                kernel[:, kernel_size // 2] = 1.0
            
            kernel = kernel / kernel.sum()
            augmented = augmented.filter(ImageFilter.Kernel(
                (kernel_size, kernel_size), 
                kernel.flatten().tolist()
            ))
        
        return augmented


def get_transforms(is_training=True):
    """
    Get transforms for data preprocessing and augmentation.
    
    Args:
        is_training (bool): Whether to include training augmentations
        
    Returns:
        transforms.Compose: Composition of transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(
    root_dir,
    annotation_file,
    batch_size=16,
    num_workers=4,
    val_split=0.2,
    clip_length=16,
    frame_interval=5
):
    """
    Create training and validation dataloaders.
    
    Args:
        root_dir (str): Directory with all the video files
        annotation_file (str): Path to the annotation CSV file
        batch_size (int): Batch size
        num_workers (int): Number of workers for dataloader
        val_split (float): Proportion of data to use for validation
        clip_length (int): Number of frames per clip
        frame_interval (int): Interval between sampled frames
        
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    # Create datasets with appropriate transforms
    train_transform = get_transforms(is_training=True)
    val_transform = get_transforms(is_training=False)
    
    # Read annotations
    annotations = pd.read_csv(annotation_file)
    
    # Split videos into train and validation sets
    videos = annotations['video_name'].unique()
    val_size = int(len(videos) * val_split)
    
    # Use a fixed seed for reproducibility
    random.seed(42)
    random.shuffle(videos)
    
    train_videos = videos[val_size:]
    val_videos = videos[:val_size]
    
    # Create filtered annotations for train and validation
    train_annotations = annotations[annotations['video_name'].isin(train_videos)]
    val_annotations = annotations[annotations['video_name'].isin(val_videos)]
    
    # Create temporary CSV files for the filtered annotations
    train_annotation_file = os.path.join(os.path.dirname(annotation_file), 'train_annotations.csv')
    val_annotation_file = os.path.join(os.path.dirname(annotation_file), 'val_annotations.csv')
    
    train_annotations.to_csv(train_annotation_file, index=False)
    val_annotations.to_csv(val_annotation_file, index=False)
    
    # Create datasets
    train_dataset = UHCTDDataset(
        root_dir=root_dir,
        annotation_file=train_annotation_file,
        transform=train_transform,
        clip_length=clip_length,
        frame_interval=frame_interval
    )
    
    val_dataset = UHCTDDataset(
        root_dir=root_dir,
        annotation_file=val_annotation_file,
        transform=val_transform,
        clip_length=clip_length,
        frame_interval=frame_interval
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Clean up temporary files
    os.remove(train_annotation_file)
    os.remove(val_annotation_file)
    
    return train_loader, val_loader


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from PIL import ImageDraw
    
    # Example paths (update with actual paths)
    root_dir = "/path/to/uhctd/videos"
    annotation_file = "/path/to/uhctd/annotations.csv"
    
    # Get transforms
    train_transform = get_transforms(is_training=True)
    
    # Create a dataset instance
    dataset = UHCTDDataset(
        root_dir=root_dir,
        annotation_file=annotation_file,
        transform=train_transform,
        clip_length=8,
        frame_interval=3
    )
    
    # Get a sample
    frames, label = dataset[0]
    
    # Print shapes and label
    print(f"Frames tensor shape: {frames.shape}")
    print(f"Label: {label}")
    
    # Create a figure to display sample frames
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.flatten()
    
    # Denormalize and convert tensor to numpy for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    frames_viz = frames * std + mean
    
    # Plot frames
    for i in range(min(8, frames.shape[1])):
        # Get frame and convert to numpy
        frame = frames_viz[:, i].permute(1, 2, 0).numpy()
        frame = np.clip(frame, 0, 1)
        
        # Display
        axes[i].imshow(frame)
        axes[i].set_title(f"Frame {i}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("sample_frames.png")
    plt.close()
    
    # Test synthetic data augmentation
    augmenter = SyntheticDataAugmentation(probability=0.8)
    
    # Convert a frame to PIL for augmentation
    frame_pil = Image.fromarray((frames_viz[:, 0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    
    # Apply augmentations
    augmented_frames = [augmenter.simulate_tampering(frame_pil) for _ in range(9)]
    
    # Display augmented frames
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, aug_frame in enumerate(augmented_frames):
        axes[i].imshow(aug_frame)
        axes[i].set_title(f"Augmentation {i+1}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("augmented_frames.png")
    plt.close()
