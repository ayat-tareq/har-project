#!/usr/bin/env python3
"""
Human Activity Recognition (HAR) Pipeline with SSL Pre-training and Evaluation
Enhanced with:
- Stratified data splitting with validation set
- Cross-user validation
- Advanced time-aware augmentations
- Hybrid CNN-Transformer architecture with fine-tuning
- Comprehensive logging and early stopping
- Ensemble methods for improved accuracy

Version 5: Fixes nested quote syntax errors within f-strings from v4.
"""

import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from utils import (
    load_data, sliding_window, SinkhornAlign, QuantumHungarian,
    evaluate_alignment, EarlyStopping, EnsemblePredictor
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, f1_score, accuracy_score,
    precision_score, recall_score, roc_curve, auc,
    classification_report
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import random
import math
import time
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("har_pipeline.log")
    ]
)
logger = logging.getLogger(__name__)

# ---------------------- Utility Functions ----------------------
def print_stage_header(stage_name, dataset):
    separator = "-" * 60
    header = f"\n{separator}\n🚀 STARTING {stage_name.upper()} - DATASET: {dataset}\n{separator}"
    print(header)
    logger.info(header)

def print_stage_footer(stage_name, dataset):
    separator = "-" * 60
    footer = f"\n{separator}\n✅ COMPLETED {stage_name.upper()} - DATASET: {dataset}\n{separator}\n"
    print(footer)
    logger.info(footer)

# ---------------------- Enhanced Model Components ----------------------
class PositionalEncoding(nn.Module):
    """Positional encoding for transformer with learnable scaling"""
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.scale = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + self.scale * self.pe[:, :x.size(1)])

class ResidualBlock(nn.Module):
    """Residual block for CNN branch"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection with projection if needed
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        
        self.activation = nn.GELU()
    
    def forward(self, x):
        residual = self.skip(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.activation(out)

class ImprovedHybridEncoder(nn.Module):
    """Improved hybrid encoder with:
    - Residual connections
    - Enhanced dropout
    - Learnable positional encoding
    - Reduced parameter count in projector
    """
    def __init__(self, input_dim, latent_dim=512, dropout=0.3, window_size=192):
        super().__init__()
        self.conv_branch = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            
            ResidualBlock(64, 128, kernel_size=5),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Reduced sequence length by 2
            nn.Dropout(dropout),
            
            ResidualBlock(128, 256, kernel_size=3),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Further reduced sequence length
        )
        
        # Calculate sequence length after conv branch
        self.seq_len_after_conv = window_size // 4  # Assuming input length of window_size, reduced by 2 twice
        
        self.pos_enc = PositionalEncoding(d_model=256, dropout=dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=1024,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=3
        )
        
        # More efficient projector with gradual dimension reduction
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * self.seq_len_after_conv, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, latent_dim)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, features, time]
        x = self.conv_branch(x)
        x = x.permute(0, 2, 1)  # [batch, time, features]
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.projector(x)

class ClassificationHead(nn.Module):
    """Classification head for fine-tuning"""
    def __init__(self, latent_dim, num_classes, dropout=0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

class EnhancedSensorAugmentation:
    """Enhanced sensor data augmentation with:
    - Time warping
    - Channel-wise dropout
    - Adaptive noise scaling
    - Frequency domain augmentations
    - Masking
    """
    def __init__(self, time_warp_limit=0.2, channel_drop_prob=0.2,
                 noise_std=0.1, scale_range=(0.8, 1.2),
                 mask_prob=0.1, freq_mask_prob=0.1):
        self.time_warp_limit = time_warp_limit
        self.channel_drop_prob = channel_drop_prob
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.mask_prob = mask_prob
        self.freq_mask_prob = freq_mask_prob
        
    def __call__(self, x):
        # Apply augmentations with probability
        if random.random() < 0.8:  # 80% chance to apply time warping
            x = self.time_warp(x)
        
        # Channel-wise dropout
        if self.channel_drop_prob > 0:
            mask = torch.ones_like(x)
            for c in range(x.shape[1]):
                if random.random() < self.channel_drop_prob:
                    mask[:, c, :] = 0
            x = x * mask
        
        # Sensor-specific scaling
        scale_factors = torch.FloatTensor(x.shape[1]).uniform_(*self.scale_range)
        x = x * scale_factors[None, :, None].to(x.device)
        
        # Adaptive noise
        noise = torch.normal(0, self.noise_std, size=x.shape, device=x.device)
        x = x + noise
        
        # Random masking (time segments)
        if random.random() < self.mask_prob:
            x = self.apply_time_mask(x)
        
        # Frequency domain augmentation
        if random.random() < self.freq_mask_prob:
            x = self.apply_freq_mask(x)
            
        return x
    
    def time_warp(self, x):
        """Time warping that stays on device when possible"""
        orig_length = x.shape[2]
        warp_factor = 1.0 + random.uniform(-self.time_warp_limit, self.time_warp_limit)
        warped_length = max(10, int(orig_length * warp_factor))
        
        # Try to perform interpolation on device if possible
        try:
            x_resampled = F.interpolate(x, size=warped_length, mode="linear", align_corners=True)
        except RuntimeError:
            # Fallback to CPU if needed
            x_cpu = x.cpu()
            x_resampled = F.interpolate(x_cpu, size=warped_length, mode="linear", align_corners=True).to(x.device)
        
        if warped_length > orig_length:
            start = random.randint(0, warped_length - orig_length)
            return x_resampled[:, :, start:start+orig_length]
        else:
            pad_left = random.randint(0, orig_length - warped_length)
            padded = torch.zeros_like(x)
            padded[:, :, pad_left:pad_left+warped_length] = x_resampled
            return padded
    
    def apply_time_mask(self, x):
        """Apply random masking in time dimension"""
        time_len = x.shape[2]
        mask_length = random.randint(time_len // 10, time_len // 4)
        mask_start = random.randint(0, time_len - mask_length)
        
        masked = x.clone()
        masked[:, :, mask_start:mask_start+mask_length] = 0
        return masked
    
    def apply_freq_mask(self, x):
        """Apply frequency domain masking"""
        # Convert to frequency domain
        x_np = x.cpu().numpy()
        for b in range(x_np.shape[0]):
            for c in range(x_np.shape[1]):
                signal = x_np[b, c]
                # FFT
                fft = np.fft.rfft(signal)
                # Mask random frequency bands
                mask_size = random.randint(len(fft) // 10, len(fft) // 5)
                mask_start = random.randint(0, len(fft) - mask_size)
                fft[mask_start:mask_start+mask_size] = 0
                # IFFT
                x_np[b, c] = np.fft.irfft(fft, n=len(signal))
        
        return torch.tensor(x_np, dtype=x.dtype, device=x.device)

class VICRegLoss(nn.Module):
    """VICReg loss function for SSL
    
    VICReg: Variance-Invariance-Covariance Regularization
    Combines invariance term with variance and covariance regularization
    """
    def __init__(self, sim_weight=25.0, var_weight=25.0, cov_weight=1.0):
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        
    def forward(self, z1, z2):
        # Invariance loss (MSE)
        sim_loss = F.mse_loss(z1, z2)
        
        # Variance loss
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
        
        # Covariance loss
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        cov_z1 = (z1.T @ z1) / (z1.shape[0] - 1)
        cov_z2 = (z2.T @ z2) / (z2.shape[0] - 1)
        
        # Remove diagonal (variances)
        mask = ~torch.eye(z1.shape[1], dtype=torch.bool, device=z1.device)
        cov_loss = cov_z1[mask].pow_(2).mean() + cov_z2[mask].pow_(2).mean()
        
        # Combine losses
        loss = self.sim_weight * sim_loss + self.var_weight * var_loss + self.cov_weight * cov_loss
        return loss

class ContrastiveLoss(nn.Module):
    """Normalized temperature-scaled cross entropy loss (NT-Xent)"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        logits = torch.mm(z1, z2.T) / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        loss = self.criterion(logits, labels)
        return loss

# ---------------------- Training & Evaluation ----------------------
def get_encoded_features(model, X, device, batch_size=256):
    """Process data in batches to avoid memory issues"""
    model.eval()
    Z_all = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            Z = model(batch)
            Z_all.append(Z.cpu())
    return torch.cat(Z_all).numpy()

def train_ssl(X_train, X_val=None, y_val=None, args=None):
    """SSL pre-training with enhanced augmentations and early stopping"""
    print_stage_header("SSL PRETRAINING", args.dataset)
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Ensure data is 3D (windowed)
    if len(X_train.shape) != 3:
        raise ValueError(f"X_train must be 3D (windowed) for SSL. Got shape: {X_train.shape}")
    
    # Create validation data if provided
    has_validation = X_val is not None and y_val is not None
    if has_validation:
        if len(X_val.shape) != 3:
             raise ValueError(f"X_val must be 3D (windowed) for validation. Got shape: {X_val.shape}")
        if len(X_val) != len(y_val):
            raise ValueError(f"X_val and y_val must have the same length. Got: {len(X_val)} and {len(y_val)}")
        X_val_win, y_val_win = X_val, y_val # Already windowed
        logger.info(f"Using validation data: X_val={X_val_win.shape}, y_val={len(y_val_win)}")
    
    # Initialize model, optimizer, and loss
    model = ImprovedHybridEncoder(X_train.shape[2], args.latent_dim, dropout=args.dropout, window_size=args.window_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Choose loss function based on args
    if args.ssl_loss == "vicreg":
        ssl_loss = VICRegLoss(sim_weight=args.sim_weight,
                             var_weight=args.var_weight,
                             cov_weight=args.cov_weight)
        logger.info("Using VICReg loss")
    else:
        ssl_loss = ContrastiveLoss(temperature=args.temperature)
        logger.info(f"Using Contrastive loss with temperature={args.temperature}")
    
    # Setup mixed precision
    use_amp = device.type in ["cuda", "mps"]
    scaler = GradScaler(enabled=use_amp)
    
    # Setup data loaders
    augmenter = EnhancedSensorAugmentation(
        time_warp_limit=args.time_warp_limit,
        channel_drop_prob=args.channel_drop_prob,
        noise_std=args.noise_std,
        scale_range=(args.scale_min, args.scale_max),
        mask_prob=args.mask_prob
    )
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Setup early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode="min")
    best_model_path = f"best_ssl_encoder_{args.dataset.lower()}.pth"
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch in pbar:
                x = batch[0].to(device)
                
                # Generate augmented views
                with torch.no_grad():
                    x1 = augmenter(x)
                    x2 = augmenter(x)
                
                # Forward pass with mixed precision
                with autocast(enabled=use_amp):
                    z1 = model(x1)
                    z2 = model(x2)
                    loss = ssl_loss(z1, z2)

                # Backward pass
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                # Update progress bar
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Calculate average loss
        avg_loss = total_loss / len(train_loader)
        epoch_log = f"[Epoch {epoch+1}/{args.epochs}] Loss: {avg_loss:.4f}"
        logger.info(epoch_log)
        
        # Validation if available
        if has_validation:
            val_acc = validate_ssl_encoder(model, X_val_win, y_val_win, device)
            logger.info(f"Validation accuracy: {val_acc*100:.2f}%")
            
            # Early stopping check
            improved = early_stopping(epoch, -val_acc)  # Negative because we want to maximize accuracy
            if improved:
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"Saved best model at epoch {epoch+1} with validation accuracy {val_acc*100:.2f}%")
            
            if early_stopping.should_stop:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        else:
            # Save model periodically if no validation
            if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                torch.save(model.state_dict(), f"ssl_encoder_{args.dataset.lower()}_epoch{epoch+1}.pth")
        
        scheduler.step()
    
    # Load best model if validation was used
    if has_validation and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        logger.info(f"Loaded best model from epoch {early_stopping.best_epoch + 1}")
    
    # Save final model
    final_path = f"ssl_encoder_{args.dataset.lower()}.pth"
    torch.save(model.state_dict(), final_path)
    save_msg = f"Saved SSL encoder to {final_path}"
    logger.info(save_msg)
    
    print_stage_footer("SSL PRETRAINING", args.dataset)
    return model

def validate_ssl_encoder(model, X_val, y_val, device):
    """Quick validation of SSL encoder using linear probe on validation set"""
    model.eval()
    
    # Extract features
    Z_val = get_encoded_features(model, X_val, device)
    
    # Verify shapes match
    if Z_val.shape[0] != len(y_val):
        logger.error(f"Shape mismatch: Z_val={Z_val.shape[0]}, y_val={len(y_val)}")
        raise AssertionError(f"Mismatch: Z_val={Z_val.shape[0]}, y_val={len(y_val)}")
    
    # Simple linear classifier
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Z_val, y_val)
    
    # Predict and calculate accuracy
    y_pred = clf.predict(Z_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    return accuracy

def fine_tune(encoder, X_train, y_train, X_val, y_val, args):
    """Fine-tune the encoder with a classification head"""
    print_stage_header("FINE TUNING", args.dataset)
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Ensure data is 3D (windowed)
    if len(X_train.shape) != 3:
        raise ValueError(f"X_train must be 3D (windowed) for fine-tuning. Got shape: {X_train.shape}")
    if len(X_val.shape) != 3:
        raise ValueError(f"X_val must be 3D (windowed) for fine-tuning. Got shape: {X_val.shape}")
    
    # Log data shapes
    logger.info(f"Training data: X_train={X_train.shape}, y_train={len(y_train)}")
    logger.info(f"Validation data: X_val={X_val.shape}, y_val={len(y_val)}")
    
    # Get number of classes
    num_classes = len(np.unique(np.concatenate([y_train, y_val])))
    
    # Create classification head
    classifier = ClassificationHead(args.latent_dim, num_classes, dropout=args.dropout).to(device)
    
    # Create full model
    class FullModel(nn.Module):
        def __init__(self, encoder, classifier):
            super().__init__()
            self.encoder = encoder
            self.classifier = classifier
        
        def forward(self, x):
            features = self.encoder(x)
            return self.classifier(features)
    
    model = FullModel(encoder, classifier).to(device)
    
    # Setup optimizer with different learning rates
    encoder_params = list(encoder.parameters())
    classifier_params = list(classifier.parameters())
    
    optimizer = optim.AdamW([
        {"params": encoder_params, "lr": args.ft_lr * 0.1},  # Lower LR for encoder
        {"params": classifier_params, "lr": args.ft_lr}       # Higher LR for classifier
    ], weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Setup data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Setup mixed precision
    use_amp = device.type in ["cuda", "mps"]
    scaler = GradScaler(enabled=use_amp)
    
    # Setup early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode="max")
    best_model_path = f"best_finetuned_{args.dataset.lower()}.pth"
    
    # Training loop
    for epoch in range(args.ft_epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.ft_epochs}") as pbar:
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass with mixed precision
                with autocast(enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                # Backward pass
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100.*correct/total:.2f}%"
                })
        
        train_acc = correct / total
        logger.info(f"[Epoch {epoch+1}] Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc*100:.2f}%")
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_acc = correct / total
        logger.info(f"[Epoch {epoch+1}] Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc*100:.2f}%")
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Early stopping check
        improved = early_stopping(epoch, val_acc)
        if improved:
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model at epoch {epoch+1} with validation accuracy {val_acc*100:.2f}%")
        
        if early_stopping.should_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        logger.info(f"Loaded best model from epoch {early_stopping.best_epoch + 1}")
    
    # Save final model
    torch.save(model.state_dict(), f"finetuned_{args.dataset.lower()}.pth")
    logger.info(f"Saved fine-tuned model to finetuned_{args.dataset.lower()}.pth")
    
    print_stage_footer("FINE TUNING", args.dataset)
    return model

def linear_probe(X_train, y_train, X_val, y_val, X_test, y_test, encoder, args):
    """Enhanced downstream evaluation with multiple classifiers and ensemble"""
    print_stage_header("LINEAR PROBING", args.dataset)
    device = get_device()
    logger.info(f"Using device: {device}")

    # Ensure data is 3D (windowed)
    if len(X_train.shape) != 3:
        raise ValueError(f"X_train must be 3D (windowed) for linear probe. Got shape: {X_train.shape}")
    if len(X_val.shape) != 3:
        raise ValueError(f"X_val must be 3D (windowed) for linear probe. Got shape: {X_val.shape}")
    if len(X_test.shape) != 3:
        raise ValueError(f"X_test must be 3D (windowed) for linear probe. Got shape: {X_test.shape}")
    
    # Log data shapes
    logger.info(f"Training data: X_train={X_train.shape}, y_train={len(y_train)}")
    logger.info(f"Validation data: X_val={X_val.shape}, y_val={len(y_val)}")
    logger.info(f"Test data: X_test={X_test.shape}, y_test={len(y_test)}")
    
    # Class distribution logging
    logger.info(f"Train class distribution: {np.bincount(y_train)}")
    logger.info(f"Val class distribution: {np.bincount(y_val)}")
    logger.info(f"Test class distribution: {np.bincount(y_test)}")

    # Extract features using the encoder
    encoder.eval()
    logger.info("Extracting features from training data...")
    Z_train = get_encoded_features(model=encoder, X=X_train, device=device)
    logger.info("Extracting features from validation data...")
    Z_val = get_encoded_features(model=encoder, X=X_val, device=device)
    logger.info("Extracting features from test data...")
    Z_test = get_encoded_features(model=encoder, X=X_test, device=device)

    # Train multiple classifiers
    classifiers = []
    
    # 1. Logistic Regression with hyperparameter tuning
    logger.info("Training Logistic Regression classifier...")
    lr_params = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["liblinear", "saga"],
        "penalty": ["l1", "l2"]
    }
    lr_clf = GridSearchCV(
        LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42),
        lr_params,
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1
    )
    lr_clf.fit(Z_train, y_train)
    logger.info(f"Best LogisticRegression params: {lr_clf.best_params_}")
    classifiers.append(lr_clf.best_estimator_)
    
    # 2. SVM
    logger.info("Training SVM classifier...")
    svm_clf = SVC(probability=True, class_weight="balanced", random_state=42)
    svm_clf.fit(Z_train, y_train)
    classifiers.append(svm_clf)
    
    # 3. Random Forest
    logger.info("Training Random Forest classifier...")
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf_clf.fit(Z_train, y_train)
    classifiers.append(rf_clf)
    
    # 4. Gradient Boosting
    logger.info("Training Gradient Boosting classifier...")
    gb_clf = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb_clf.fit(Z_train, y_train)
    classifiers.append(gb_clf)
    
    # Evaluate individual classifiers on validation set
    val_scores = []
    for i, clf in enumerate(classifiers):
        y_pred_val = clf.predict(Z_val)
        val_acc = accuracy_score(y_val, y_pred_val)
        # FIX: Corrected nested quotes in f-string
        val_f1 = f1_score(y_val, y_pred_val, average='weighted')
        val_scores.append((val_acc, val_f1))
        logger.info(f"Classifier {i+1} - Val Accuracy: {val_acc*100:.2f}%, F1: {val_f1:.4f}")
    
    # Create ensemble with weights based on validation performance
    weights = [score[1] for score in val_scores]  # Use F1 scores as weights
    weights = np.array(weights) / sum(weights)  # Normalize
    ensemble = EnsemblePredictor(classifiers, weights)
    
    # Save best individual classifier and ensemble
    best_idx = np.argmax([score[1] for score in val_scores])
    best_clf = classifiers[best_idx]
    joblib.dump(best_clf, f"best_classifier_{args.dataset.lower()}.pkl")
    joblib.dump(ensemble, f"ensemble_classifier_{args.dataset.lower()}.pkl")
    logger.info(f"Saved best classifier and ensemble to classifier_{args.dataset.lower()}.pkl")

    # Evaluate on test set
    # Best individual classifier
    y_pred_test = best_clf.predict(Z_test)
    y_proba_test = best_clf.predict_proba(Z_test)
    
    # Ensemble
    y_pred_ensemble = ensemble.predict(Z_test)
    
    # Metrics calculation
    def print_metrics(y_true, y_pred, name):
        # FIX: Corrected nested quotes in f-strings
        metrics = [
            f"Accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%",
            f"F1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}",
            f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}",
            f"Recall: {recall_score(y_true, y_pred, average='weighted'):.4f}"
        ]
        
        print(f"\n📊 {name} METRICS:")
        print("\n".join([f"| {m}" for m in metrics]))
        print("\n📝 Classification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        
        # Log metrics
        logger.info(f"{name} METRICS:")
        for m in metrics:
            logger.info(m)
        logger.info("Classification Report:\n" + classification_report(y_true, y_pred, zero_division=0))

    print_metrics(y_test, y_pred_test, "BEST CLASSIFIER")
    print_metrics(y_test, y_pred_ensemble, "ENSEMBLE")

    # Confusion Matrix for ensemble
    classes = np.unique(np.concatenate([y_train, y_val, y_test]))
    cm = confusion_matrix(y_test, y_pred_ensemble, labels=classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix (Ensemble) - {args.dataset}", fontsize=14)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(f"confusion_matrix_{args.dataset.lower()}.png", dpi=300)
    logger.info(f"Saved confusion matrix to confusion_matrix_{args.dataset.lower()}.png")

    # ROC Curve (only for binary/multiclass)
    if len(classes) > 1:
        y_test_bin = label_binarize(y_test, classes=classes)
        n_classes = y_test_bin.shape[1]
        
        fpr, tpr, roc_auc = {}, {}, {}
        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba_test[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=2,
                     label=f"Class {classes[i]} (AUC = {roc_auc[i]:.2f})")

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves - {args.dataset}", fontsize=14)
        plt.legend(loc="lower right")
        plt.savefig(f"roc_curves_{args.dataset.lower()}.png", dpi=300)
        logger.info(f"Saved ROC curves to roc_curves_{args.dataset.lower()}.png")

    # t-SNE Visualization
    tsne = TSNE(n_components=2, perplexity=40, n_iter=3000, random_state=42)
    Z_combined = np.vstack([Z_train, Z_val, Z_test])
    y_combined = np.concatenate([y_train, y_val, y_test])
    Z_2d = tsne.fit_transform(Z_combined)
    
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(Z_2d[:, 0], Z_2d[:, 1],
                         c=y_combined,
                         cmap="viridis", s=25, alpha=0.6)
    plt.colorbar(scatter, ticks=np.unique(y_test))
    plt.title(f"t-SNE Embeddings - {args.dataset}", fontsize=16)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig(f"tsne_embeddings_{args.dataset.lower()}.png", dpi=300)
    logger.info(f"Saved t-SNE plot to tsne_embeddings_{args.dataset.lower()}.png")

    print_stage_footer("LINEAR PROBING", args.dataset)
    return ensemble

def evaluate_alignment_with_mh(X, y, user_ids, encoder, args):
    """Cross-user alignment evaluation with improved time-aware matching"""
    print_stage_header("ALIGNMENT EVALUATION", args.dataset)
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Ensure data is 3D (windowed)
    if len(X.shape) != 3:
        raise ValueError(f"X must be 3D (windowed) for alignment evaluation. Got shape: {X.shape}")
    
    # Log data shapes
    logger.info(f"Data: X={X.shape}, y={len(y)}, user_ids={len(user_ids)}")

    encoder.eval()
    logger.info("Extracting features...")
    Z = get_encoded_features(model=encoder, X=X, device=device)

    # Cross-user validation with multiple splits
    unique_users = np.unique(user_ids)
    logger.info(f"Found {len(unique_users)} unique users")
    
    # Perform multiple evaluations with different user splits
    n_splits = min(5, len(unique_users) // 2)
    accuracies = []
    f1_scores = []
    temporal_coherences = []
    
    for split in range(n_splits):
        # Randomly split users
        np.random.shuffle(unique_users)
        split_idx = len(unique_users) // 2
        source_users = unique_users[:split_idx]
        target_users = unique_users[split_idx:]
        
        source_mask = np.isin(user_ids, source_users)
        target_mask = np.isin(user_ids, target_users)
        
        Z_source = Z[source_mask]
        Z_target = Z[target_mask]
        y_source = y[source_mask]
        y_target = y[target_mask]
        
        # Try both alignment methods
        aligners = [
            ("QuantumHungarian", QuantumHungarian(temp=args.temp, feature_weight=args.feature_weight)),
            ("SinkhornAlign", SinkhornAlign(n_iter=args.n_iter, epsilon=args.epsilon))
        ]
        
        for name, aligner in aligners:
            row_ind, col_ind = aligner(Z_source, Z_target)
            
            # Evaluate alignment with enhanced metrics
            acc, f1, temporal = evaluate_alignment(y_source, y_target, row_ind, col_ind)
            
            logger.info(f"Split {split+1}/{n_splits}, {name} - "
                       f"Accuracy: {acc*100:.2f}%, F1: {f1:.4f}, "
                       f"Temporal Coherence: {temporal:.4f}")
            
            accuracies.append(acc)
            f1_scores.append(f1)
            temporal_coherences.append(temporal)
    
    # Report average results
    avg_acc = np.mean(accuracies)
    avg_f1 = np.mean(f1_scores)
    avg_temporal = np.mean(temporal_coherences)
    
    results_msg = (f"\n🎯 Cross-User Alignment Results (Avg over {n_splits} splits):\n"
                  f"Accuracy: {avg_acc*100:.2f}%\n"
                  f"F1 Score: {avg_f1:.4f}\n"
                  f"Temporal Coherence: {avg_temporal:.4f}")
    
    print(results_msg)
    logger.info(results_msg)
    print_stage_footer("ALIGNMENT EVALUATION", args.dataset)
    
    return avg_acc, avg_f1, avg_temporal

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# ---------------------- Main Function ----------------------
def main():
    parser = argparse.ArgumentParser(description="HAR SSL Pipeline with Enhanced Accuracy")
    
    # Dataset and mode
    parser.add_argument("--mode", type=str, default="pretrain",
                        choices=["pretrain", "finetune", "linear_probe", "alignment", "all"],
                        help="Pipeline mode")
    parser.add_argument("--dataset", type=str, default="PAMAP2",
                        choices=["UCI", "PAMAP2"],
                        help="Dataset to use")
    
    # Model parameters
    parser.add_argument("--latent_dim", type=int, default=512,
                        help="Latent dimension size")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate")
    
    # Window parameters
    parser.add_argument("--window_size", type=int, default=192,
                        help="Window size for sliding window")
    parser.add_argument("--window_step", type=int, default=64,
                        help="Step size for sliding window")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of epochs for SSL pretraining")
    parser.add_argument("--ft_epochs", type=int, default=100,
                        help="Number of epochs for fine-tuning")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="Learning rate for SSL pretraining")
    parser.add_argument("--ft_lr", type=float, default=0.0005,
                        help="Learning rate for fine-tuning")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--patience", type=int, default=15,
                        help="Patience for early stopping")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    
    # SSL parameters
    parser.add_argument("--ssl_loss", type=str, default="contrastive",
                        choices=["contrastive", "vicreg"],
                        help="SSL loss function")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for contrastive loss")
    parser.add_argument("--sim_weight", type=float, default=25.0,
                        help="Similarity weight for VICReg loss")
    parser.add_argument("--var_weight", type=float, default=25.0,
                        help="Variance weight for VICReg loss")
    parser.add_argument("--cov_weight", type=float, default=1.0,
                        help="Covariance weight for VICReg loss")
    
    # Augmentation parameters
    parser.add_argument("--time_warp_limit", type=float, default=0.2,
                        help="Time warping limit")
    parser.add_argument("--channel_drop_prob", type=float, default=0.2,
                        help="Channel dropout probability")
    parser.add_argument("--noise_std", type=float, default=0.1,
                        help="Noise standard deviation")
    parser.add_argument("--scale_min", type=float, default=0.8,
                        help="Minimum scaling factor")
    parser.add_argument("--scale_max", type=float, default=1.2,
                        help="Maximum scaling factor")
    parser.add_argument("--mask_prob", type=float, default=0.1,
                        help="Masking probability")
    
    # Alignment parameters
    parser.add_argument("--temp", type=float, default=0.2,
                        help="Temperature for QuantumHungarian")
    parser.add_argument("--feature_weight", type=float, default=0.7,
                        help="Feature weight for QuantumHungarian")
    parser.add_argument("--n_iter", type=int, default=20,
                        help="Number of iterations for SinkhornAlign")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Epsilon for SinkhornAlign")
    
    args = parser.parse_args()
    
    # Print configuration
    logger.info(f"Starting pipeline with arguments: {vars(args)}")
    
    # Load dataset
    if args.dataset == "UCI":
        X_train_raw, y_train_raw, X_test_raw, y_test_raw, user_train_raw, user_test_raw = load_data(args.dataset)
        # Create validation set from training data
        X_train_raw, X_val_raw, y_train_raw, y_val_raw, user_train_raw, user_val_raw = train_test_split(
            X_train_raw, y_train_raw, user_train_raw, test_size=0.2, stratify=y_train_raw, random_state=42
        )
        
        # Apply sliding window to UCI data immediately after loading/splitting
        logger.info("Applying sliding window to UCI data...")
        X_train, y_train, user_train = sliding_window(X_train_raw, args.window_size, args.window_step, y_train_raw, user_train_raw)
        X_val, y_val, user_val = sliding_window(X_val_raw, args.window_size, args.window_step, y_val_raw, user_val_raw)
        X_test, y_test, user_test = sliding_window(X_test_raw, args.window_size, args.window_step, y_test_raw, user_test_raw)
        logger.info(f"UCI data after windowing: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
        
    else:  # PAMAP2 (already windowed during loading)
        X_train, y_train, X_val, y_val, X_test, y_test, user_train, user_val, user_test = load_data(args.dataset)
    
    # Ensure data is 3D before proceeding
    if len(X_train.shape) != 3:
        raise ValueError(f"X_train must be 3D after loading/windowing. Got shape: {X_train.shape}")
    
    # Execute pipeline based on mode
    if args.mode == "pretrain" or args.mode == "all":
        encoder = train_ssl(X_train, X_val, y_val, args)
    else:
        # Load pre-trained encoder
        encoder = ImprovedHybridEncoder(X_train.shape[2], args.latent_dim, window_size=args.window_size).to(get_device())
        try:
            encoder.load_state_dict(torch.load(f"ssl_encoder_{args.dataset.lower()}.pth", map_location=get_device()))
            logger.info("Loaded pre-trained encoder successfully")
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise
    
    if args.mode == "finetune" or args.mode == "all":
        fine_tuned = fine_tune(encoder, X_train, y_train, X_val, y_val, args)
    
    if args.mode == "linear_probe" or args.mode == "all":
        ensemble = linear_probe(X_train, y_train, X_val, y_val, X_test, y_test, encoder, args)
    
    if args.mode == "alignment" or args.mode == "all":
        # Combine all data for alignment evaluation
        X_all = np.concatenate([X_train, X_val, X_test])
        y_all = np.concatenate([y_train, y_val, y_test])
        user_all = np.concatenate([user_train, user_val, user_test])
        evaluate_alignment_with_mh(X_all, y_all, user_all, encoder, args)
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")

