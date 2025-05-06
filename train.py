import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# Import the model architecture and data processing
from model import SelfGuidingSentinels
from data_loader import UHCTDDataset, create_dataloaders, get_transforms

class Trainer:
    """
    Trainer class for Self-Guiding Sentinels model
    """
    def __init__(self, args):
        """
        Initialize the trainer with arguments
        
        Args:
            args: Command line arguments
        """
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.start_epoch = 0
        
        # Set random seeds for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Initialize model
        self._init_model()
        
        # Initialize data loaders
        self._init_dataloaders()
        
        # Initialize optimizer and scheduler
        self._init_training_components()
        
        # Initialize metrics tracking
        self._init_metrics()
        
    def _init_model(self):
        """Initialize the Self-Guiding Sentinels model"""
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
            dropout=self.args.dropout
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # If resuming training, load checkpoint
        if self.args.resume:
            self._load_checkpoint()
    
    def _init_dataloaders(self):
        """Initialize data loaders for training and validation"""
        print("Creating data loaders...")
        self.train_loader, self.val_loader = create_dataloaders(
            root_dir=self.args.data_root,
            annotation_file=self.args.annotation_file,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            val_split=self.args.val_split,
            clip_length=self.args.clip_length,
            frame_interval=self.args.frame_interval
        )
        print(f"Created data loaders. Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
    
    def _init_training_components(self):
        """Initialize optimizer, scheduler, loss function, and gradient scaler"""
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epochs,
            eta_min=self.args.min_lr
        )
        
        # Loss function
        class_weights = None
        if self.args.use_class_weights:
            # Calculate class weights from the dataset (inverse frequency)
            labels = []
            for _, label in self.train_loader.dataset:
                labels.append(label)
            labels = torch.tensor(labels)
            class_counts = torch.bincount(labels)
            class_weights = 1.0 / class_counts.float()
            class_weights = class_weights / class_weights.sum() * len(class_counts)
            class_weights = class_weights.to(self.device)
            
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Gradient scaler for mixed precision training
        self.scaler = GradScaler() if self.args.mixed_precision else None
        
        # Load optimizer state if resuming
        if self.args.resume and 'optimizer' in self.checkpoint:
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            self.scheduler.load_state_dict(self.checkpoint['scheduler'])
            if self.scaler and 'scaler' in self.checkpoint:
                self.scaler.load_state_dict(self.checkpoint['scaler'])
    
    def _init_metrics(self):
        """Initialize metrics for tracking training and validation performance"""
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        
    def _load_checkpoint(self):
        """Load checkpoint from file"""
        print(f"Loading checkpoint from {self.args.resume}")
        self.checkpoint = torch.load(self.args.resume, map_location=self.device)
        self.model.load_state_dict(self.checkpoint['model'])
        self.start_epoch = self.checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {self.start_epoch}")
        
        # Load metrics history if available
        if 'metrics' in self.checkpoint:
            metrics = self.checkpoint['metrics']
            self.train_losses = metrics.get('train_losses', [])
            self.val_losses = metrics.get('val_losses', [])
            self.train_accs = metrics.get('train_accs', [])
            self.val_accs = metrics.get('val_accs', [])
            self.best_val_acc = metrics.get('best_val_acc', 0.0)
            self.best_val_loss = metrics.get('best_val_loss', float('inf'))
            self.best_val_f1 = metrics.get('best_val_f1', 0.0)
    
    def save_checkpoint(self, epoch, filename=None, is_best=False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            filename: Optional filename for the checkpoint
            is_best: Whether this is the best model so far
        """
        if filename is None:
            filename = os.path.join(self.args.output_dir, f'checkpoint_epoch_{epoch}.pth')
        
        # Prepare metrics
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1
        }
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'metrics': metrics,
            'args': self.args
        }
        
        # Add scaler state if using mixed precision
        if self.scaler:
            checkpoint['scaler'] = self.scaler.state_dict()
        
        # Save the checkpoint
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")
        
        # Save best model separately
        if is_best:
            best_path = os.path.join(self.args.output_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")
    
    def train_epoch(self, epoch):
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            train_loss, train_acc: Average loss and accuracy for the epoch
        """
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} [Train]")
        
        for batch_idx, (data, targets) in enumerate(pbar):
            # Move data to device
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Reset gradients
            self.optimizer.zero_grad()
            
            if self.args.mixed_precision:
                # Mixed precision training
                with autocast():
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
                
                # Scale gradients and optimize
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            avg_loss = train_loss / (batch_idx + 1)
            acc = 100.0 * correct / total
            pbar.set_postfix({'loss': f"{avg_loss:.4f}", 'acc': f"{acc:.2f}%"})
        
        # Calculate average metrics
        train_loss /= len(self.train_loader)
        train_acc = 100.0 * correct / total
        
        # Update learning rate
        self.scheduler.step()
        
        return train_loss, train_acc
    
    def validate(self):
        """
        Validate the model on the validation set
        
        Returns:
            val_loss, val_acc, val_f1: Validation loss, accuracy, and F1 score
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        # For metrics calculation
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for batch_idx, (data, targets) in enumerate(pbar):
                # Move data to device
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                # Update metrics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Store predictions and targets for metrics
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Update progress bar
                avg_loss = val_loss / (batch_idx + 1)
                acc = 100.0 * correct / total
                pbar.set_postfix({'loss': f"{avg_loss:.4f}", 'acc': f"{acc:.2f}%"})
        
        # Calculate average metrics
        val_loss /= len(self.val_loader)
        val_acc = 100.0 * correct / total
        
        # Calculate F1 score
        val_f1 = f1_score(all_targets, all_preds, average='macro')
        
        # Calculate additional metrics
        precision = precision_score(all_targets, all_preds, average='macro')
        recall = recall_score(all_targets, all_preds, average='macro')
        
        # Create confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        # Calculate ROC AUC if it's a multi-class problem
        if self.args.n_classes > 2:
            # Convert to one-hot encoding for multi-class
            y_true_oh = np.zeros((len(all_targets), self.args.n_classes))
            y_true_oh[np.arange(len(all_targets)), all_targets] = 1
            
            # Get probability outputs
            all_probs = []
            with torch.no_grad():
                for data, _ in self.val_loader:
                    data = data.to(self.device)
                    outputs = self.model(data)
                    probs = torch.softmax(outputs, dim=1)
                    all_probs.extend(probs.cpu().numpy())
            
            # Calculate ROC AUC
            roc_auc = roc_auc_score(y_true_oh, all_probs, multi_class='ovr')
        else:
            # Binary classification
            all_probs = []
            with torch.no_grad():
                for data, _ in self.val_loader:
                    data = data.to(self.device)
                    outputs = self.model(data)
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    all_probs.extend(probs.cpu().numpy())
            
            roc_auc = roc_auc_score(all_targets, all_probs)
        
        # Print detailed metrics
        print(f"\nValidation Results:")
        print(f"Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        print(f"F1 Score: {val_f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        
        # Save confusion matrix as figure
        plt.figure(figsize=(10, 8))
        class_names = [f"Class {i}" for i in range(self.args.n_classes)]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.output_dir, f"confusion_matrix_epoch_{self.start_epoch}.png"))
        plt.close()
        
        return val_loss, val_acc, val_f1
    
    def plot_metrics(self):
        """Plot and save training and validation metrics"""
        plt.figure(figsize=(12, 5))
        
        # Plot training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curves')
        
        # Plot training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train')
        plt.plot(self.val_accs, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Accuracy Curves')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.output_dir, 'training_metrics.png'))
        plt.close()
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'epoch': list(range(1, len(self.train_losses) + 1)),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_acc': self.train_accs,
            'val_acc': self.val_accs
        })
        metrics_df.to_csv(os.path.join(self.args.output_dir, 'training_metrics.csv'), index=False)
    
    def train(self):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Start timing
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_f1 = self.validate()
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Check if this is the best model
            is_best = False
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                is_best = True
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
            
            # Save checkpoint
            self.save_checkpoint(
                epoch=epoch,
                filename=os.path.join(self.args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                is_best=is_best
            )
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{self.args.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                  f"Val F1: {val_f1:.4f}")
            
            # Plot metrics after each epoch
            self.plot_metrics()
            
            # Learning rate info
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Current Learning Rate: {current_lr:.6f}")
        
        # Training complete
        total_time = time.time() - start_time
        print(f"Training complete in {total_time/60:.2f} minutes")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print(f"Best Validation F1 Score: {self.best_val_f1:.4f}")


def parse_args():
    """
    Parse command line arguments
    
    Returns:
        args: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train Self-Guiding Sentinels model")
    
    # Data parameters
    parser.add_argument('--data-root', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--annotation-file', type=str, required=True, help='Path to annotation CSV file')
    parser.add_argument('--output-dir', type=str, default='output', help='Path to save outputs')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation set ratio')
    
    # Model parameters
    parser.add_argument('--img-size', type=int, default=224, help='Input image size')
    parser.add_argument('--patch-size', type=int, default=16, help='Patch size for ViT')
    parser.add_argument('--in-channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--embed-dim', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--vit-depth', type=int, default=12, help='Depth of ViT')
    parser.add_argument('--n-heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--mlp-ratio', type=float, default=4.0, help='MLP ratio')
    parser.add_argument('--diffusion-hidden-dim', type=int, default=512, help='Hidden dim for diffusion component')
    parser.add_argument('--n-classes', type=int, default=4, help='Number of classes')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-5, help='Minimum learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--clip-length', type=int, default=16, help='Number of frames per clip')
    parser.add_argument('--frame-interval', type=int, default=5, help='Interval between sampled frames')
    parser.add_argument('--use-class-weights', action='store_true', help='Use class weights in loss function')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--mixed-precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (cuda:0, cuda:1, cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    return args


def main():
    """Main function"""
    args = parse_args()
    
    # Print args
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    # Initialize trainer
    trainer = Trainer(args)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
