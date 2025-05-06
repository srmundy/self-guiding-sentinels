import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc, 
    precision_recall_curve,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)
import cv2
from PIL import Image
import torchvision.transforms as transforms
import time

# Import your model and data loader
from model import SelfGuidingSentinels
from data_loader import UHCTDDataset, get_transforms

class SentinelEvaluator:
    """
    Evaluator for Self-Guiding Sentinels model on test data
    """
    def __init__(self, args):
        """
        Initialize the evaluator
        
        Args:
            args: Command line arguments
        """
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        
        # Set random seeds for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load the model
        self._load_model()
        
        # Create test dataset and loader
        self._create_dataloader()
        
        # Class names mapping
        self.class_names = {
            0: "Normal Operation",
            1: "Environmental Factors",
            2: "Legitimate Maintenance",
            3: "Physical Attack"
        }
    
    def _load_model(self):
        """Load the pre-trained model"""
        print(f"Loading model from {self.args.model_path}")
        
        # Initialize model architecture
        self.model = SelfGuidingSentinels(
            img_size=self.args.img_size,
            patch_size=self.args.patch_size,
            in_channels=self.args.in_channels,
            embed_dim=self.args.embed_dim,
            vit_depth=self.args.vit_depth,
            n_heads=self.args.n_heads,
            mlp_ratio=self.args.mlp_ratio,
            diffusion_hidden_dim=self.args.diffusion_hidden_dim,
            n_classes=self.args.n_classes,
            dropout=0.0  # No dropout during evaluation
        )
        
        # Load weights
        checkpoint = torch.load(self.args.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set to evaluation mode
        self.model.eval()
        
        print("Model loaded successfully")
    
    def _create_dataloader(self):
        """Create test dataloader"""
        # Get test transforms
        test_transforms = get_transforms(is_training=False)
        
        # Create test dataset
        self.test_dataset = UHCTDDataset(
            root_dir=self.args.data_root,
            annotation_file=self.args.annotation_file,
            transform=test_transforms,
            clip_length=self.args.clip_length,
            frame_interval=self.args.frame_interval
        )
        
        # Create test dataloader
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        print(f"Test dataset created with {len(self.test_dataset)} samples")
    
    def evaluate(self):
        """
        Evaluate the model on the test set
        
        Returns:
            metrics: Dictionary containing evaluation metrics
        """
        print("Starting evaluation...")
        
        # Track predictions and ground truth
        all_preds = []
        all_targets = []
        all_probs = []
        
        # Measure inference time
        total_inference_time = 0
        num_samples = 0
        
        with torch.no_grad():
            for data, targets in tqdm(self.test_loader, desc="Evaluating"):
                # Move data to device
                data = data.to(self.device)
                batch_size = data.size(0)
                
                # Measure inference time
                start_time = time.time()
                
                # Forward pass
                outputs = self.model(data)
                probs = torch.softmax(outputs, dim=1)
                
                # Measure elapsed time
                elapsed_time = time.time() - start_time
                total_inference_time += elapsed_time
                num_samples += batch_size
                
                # Get predictions
                _, preds = torch.max(outputs, 1)
                
                # Store results
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # Calculate average inference time
        avg_inference_time = (total_inference_time / num_samples) * 1000  # ms
        print(f"Average inference time: {avg_inference_time:.2f} ms per sample")
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='macro')
        recall = recall_score(all_targets, all_preds, average='macro')
        f1 = f1_score(all_targets, all_preds, average='macro')
        
        # Calculate class-specific metrics
        class_precision = precision_score(all_targets, all_preds, average=None)
        class_recall = recall_score(all_targets, all_preds, average=None)
        class_f1 = f1_score(all_targets, all_preds, average=None)
        
        # Calculate ROC AUC
        if self.args.n_classes == 2:
            # Binary classification
            roc_auc = roc_auc_score(all_targets, all_probs[:, 1])
        else:
            # Multi-class classification (one-vs-rest)
            # Convert targets to one-hot encoding
            targets_one_hot = np.zeros((len(all_targets), self.args.n_classes))
            targets_one_hot[np.arange(len(all_targets)), all_targets] = 1
            roc_auc = roc_auc_score(targets_one_hot, all_probs, multi_class='ovr')
        
        # Print overall metrics
        print("\nEvaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (macro): {precision:.4f}")
        print(f"Recall (macro): {recall:.4f}")
        print(f"F1 Score (macro): {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Average Inference Time:
