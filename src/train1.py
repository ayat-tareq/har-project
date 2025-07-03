#!/usr/bin/env python3
"""
Enhanced HAR SSL Pipeline (v8) with Improved Stability & Performance
- Fixed SWA/MPS compatibility
- Enhanced feature scaling for linear probing
- Added channel masking augmentation
- Improved model architecture
- Better handling of class imbalance
"""
import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR
from utils import (
    load_data, sliding_window, SinkhornAlign, QuantumHungarian,
    evaluate_alignment, EarlyStopping, EnsemblePredictor
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, f1_score, accuracy_score,
    precision_score, recall_score, roc_curve, auc,
    classification_report
)
from sklearn.preprocessing import label_binarize, LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import random
import math
import time
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

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

# --- Constants for fixed parameters ---
ALIGNMENT_TEMP = 0.2
ALIGNMENT_N_ITER = 100

# Global variable to store label encoder if subset is used
subset_label_encoder = None

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

def save_metrics_to_file(metrics_dict, filename):
    try:
        with open(filename, "w") as f:
            for key, value in metrics_dict.items():
                f.write(f"{key}: {value}\n")
        logger.info(f"Metrics saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save metrics to {filename}: {e}")

# ---------------------- Enhanced Model Components ----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.scale = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + self.scale * self.pe[:, :x.size(1)])

class ChannelAttention(nn.Module):
    """Channel attention module"""
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.shape
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        scale = torch.sigmoid(avg_out + max_out)
        return x * scale.view(b, c, 1)

class EnhancedHybridEncoder(nn.Module):
    """Enhanced encoder with depthwise separable convolutions and channel attention"""
    def __init__(self, input_dim, latent_dim=128, dropout=0.3, transformer_seq_len=32, num_transformer_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.transformer_seq_len = transformer_seq_len
        
        # Calculate divisible output channels
        conv1_out = 64
        if input_dim > 0 and conv1_out % input_dim != 0:
            conv1_out = (conv1_out // input_dim + 1) * input_dim
        
        # First depthwise separable convolution
        self.depthwise1 = nn.Conv1d(input_dim, input_dim, kernel_size=5, padding=2, groups=input_dim)
        self.pointwise1 = nn.Conv1d(input_dim, conv1_out, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(conv1_out)
        
        # Second depthwise separable convolution
        self.depthwise2 = nn.Conv1d(conv1_out, conv1_out, kernel_size=3, padding=1, groups=conv1_out)
        self.pointwise2 = nn.Conv1d(conv1_out, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.attn = ChannelAttention(128)
        self.pool = nn.AdaptiveAvgPool1d(transformer_seq_len)
        
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        self.pos_enc = PositionalEncoding(d_model=128, dropout=dropout)
        
        # Transformer with layer normalization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=512,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        
        # Enhanced projector with residual connection
        self.projector = nn.Sequential(
            nn.Linear(128 * self.transformer_seq_len, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, latent_dim)
        )
        
        # Skip connection
        self.skip = nn.Linear(128 * self.transformer_seq_len, latent_dim)

    def forward(self, x):
        # Input handling - ensure proper dimensions
        if x.dim() == 3 and x.shape[2] == self.input_dim:  # [batch, seq_len, features]
            x = x.permute(0, 2, 1)  # Convert to [batch, features, seq_len]
        elif x.dim() == 3 and x.shape[1] == self.input_dim:  # [batch, features, seq_len]
            pass  # Already in correct format
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}. Expected (batch, seq_len, features) or (batch, features, seq_len)")
            
        # Convolutional branch
        x = self.depthwise1(x)
        x = self.pointwise1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        
        x = self.depthwise2(x)
        x = self.pointwise2(x)
        x = self.bn2(x)
        x = self.gelu(x)
        x = self.attn(x)
        x = self.pool(x)
        
        x = x.permute(0, 2, 1)  # [batch, seq, features]
        x = self.pos_enc(x)
        x = self.transformer(x)
        
        # Flatten and project
        x_flat = x.reshape(x.size(0), -1)
        z = self.projector(x_flat) + self.skip(x_flat)
        return z

class EnhancedClassificationHead(nn.Module):
    """Classification head with label smoothing and temperature scaling"""
    def __init__(self, latent_dim, num_classes, dropout=0.5, smoothing=0.1, temp=0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        self.smoothing = smoothing
        self.temp = nn.Parameter(torch.ones(1) * temp)
    
    def forward(self, x, targets=None):
        logits = self.classifier(x) / self.temp
        if targets is not None:
            return self.smooth_cross_entropy(logits, targets)
        return logits
    
    def smooth_cross_entropy(self, logits, labels):
        confidence = 1.0 - self.smoothing
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=labels.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# Enhanced Augmentation with Sensor-Specific Noise
class SensorSpecificAugmentation:
    def __init__(self, time_warp_limit=0.2, channel_mask_prob=0.3,
                 acc_noise_std=0.1, gyro_noise_std=0.15,
                 scale_range=(0.8, 1.2), max_mask_length=20,
                 freq_mask_prob=0.2, sensor_types=None):
        self.time_warp_limit = time_warp_limit
        self.channel_mask_prob = channel_mask_prob
        self.acc_noise_std = acc_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.scale_range = scale_range
        self.max_mask_length = max_mask_length
        self.freq_mask_prob = freq_mask_prob
        self.sensor_types = sensor_types or []
        
    def time_mask(self, x):
        """Apply time masking to sensor data"""
        if x.shape[2] < 10:
            return x
        seq_len = x.shape[2]
        mask_length = random.randint(1, self.max_mask_length)
        mask_start = random.randint(0, seq_len - mask_length)
        x_masked = x.clone()
        x_masked[:, :, mask_start:mask_start+mask_length] = 0
        return x_masked
    
    def channel_mask(self, x):
        """Apply channel-wise masking"""
        mask = torch.ones_like(x)
        for c in range(x.shape[1]):
            if random.random() < self.channel_mask_prob:
                mask[:, c, :] = 0
        return x * mask
    
    def frequency_mask(self, x):
        """Apply frequency domain masking"""
        if random.random() < self.freq_mask_prob:
            try:
                x_fft = torch.fft.rfft(x, dim=2)
                mask_len = random.randint(1, x_fft.shape[2]//2)
                mask_start = random.randint(0, x_fft.shape[2] - mask_len)
                x_fft[:, :, mask_start:mask_start+mask_len] = 0
                return torch.fft.irfft(x_fft, n=x.shape[2], dim=2)
            except Exception:
                return x
        return x
    
    def time_warp(self, x):
        if x.shape[2] < 10:
            return x
        orig_length = x.shape[2]
        warp_factor = 1.0 + random.uniform(-self.time_warp_limit, self.time_warp_limit)
        warped_length = max(10, int(orig_length * warp_factor))
        try:
            x_resampled = F.interpolate(x, size=warped_length, mode="linear", align_corners=True)
        except RuntimeError:
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

    def __call__(self, x):
        x = self.time_warp(x)
        x = self.time_mask(x)
        x = self.channel_mask(x)
        x = self.frequency_mask(x)
        
        # Sensor-specific noise
        noise = torch.zeros_like(x)
        for i, sensor_type in enumerate(self.sensor_types):
            if sensor_type == 'acc':
                noise_std = self.acc_noise_std
            elif sensor_type == 'gyro':
                noise_std = self.gyro_noise_std
            else:  # Default noise
                noise_std = (self.acc_noise_std + self.gyro_noise_std) / 2
                
            noise[:, i, :] = torch.normal(0, noise_std, size=(x.shape[0], x.shape[2]))
        
        x = x + noise
        
        # Channel-specific scaling
        scale_factors = torch.FloatTensor(x.shape[1]).uniform_(*self.scale_range)
        x = x * scale_factors[None, :, None].to(x.device)
        
        return x

class TemporalConsistencyLoss(nn.Module):
    """Encourages temporal smoothness in representations"""
    def __init__(self, weight=0.3):
        super().__init__()
        self.weight = weight
        
    def forward(self, z):
        if z.size(0) < 2:  # Need at least 2 samples
            return torch.tensor(0.0, device=z.device)
            
        z_diff = z[1:] - z[:-1]
        return self.weight * torch.mean(z_diff.pow(2))

class VICRegLoss(nn.Module):
    """Enhanced VICReg loss with learnable weights and temporal consistency"""
    def __init__(self, sim_weight=25.0, var_weight=25.0, cov_weight=1.0, temp_weight=0.3):
        super().__init__()
        self.sim_weight = nn.Parameter(torch.tensor(sim_weight))
        self.var_weight = nn.Parameter(torch.tensor(var_weight))
        self.cov_weight = nn.Parameter(torch.tensor(cov_weight))
        self.temp_loss = TemporalConsistencyLoss(weight=temp_weight)
        
    def forward(self, z1, z2):
        sim_loss = F.mse_loss(z1, z2)
        
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
        
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        cov_z1 = (z1.T @ z1) / (z1.shape[0] - 1)
        cov_z2 = (z2.T @ z2) / (z2.shape[0] - 1)
        mask = ~torch.eye(z1.shape[1], dtype=torch.bool, device=z1.device)
        cov_loss = cov_z1[mask].pow_(2).mean() + cov_z2[mask].pow_(2).mean()
        
        temporal_loss = self.temp_loss(z1) + self.temp_loss(z2)
        
        loss = (self.sim_weight * sim_loss +
                self.var_weight * var_loss +
                self.cov_weight * cov_loss +
                temporal_loss)
        return loss

class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss"""
    def __init__(self, temperature=0.1, base_temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        device = z1.device
        batch_size = z1.shape[0]
        
        # Normalize feature vectors
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z1, z2.T) / self.temperature
        
        # Create labels
        labels = torch.arange(batch_size, device=device)
        
        # Loss calculation
        loss = self.criterion(sim_matrix, labels)
        return loss

# ---------------------- Training & Evaluation ----------------------
def get_encoded_features(model, X, device, batch_size=64):
    model.eval()
    Z_all = []
    with torch.no_grad():
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32)
        else:
            X_tensor = X.float()
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch in loader:
            batch_data = batch[0].to(device)
            Z = model(batch_data)
            Z_all.append(Z.cpu())
    return torch.cat(Z_all).numpy()

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Use float32 for MPS compatibility
        torch.set_default_dtype(torch.float32)
        return torch.device("mps")
    else:
        return torch.device("cpu")

def train_ssl(X_train, X_val=None, y_val=None, args=None):
    print_stage_header("SSL PRETRAINING", args.dataset)
    device = get_device()
    logger.info(f"Using device: {device}")
    
    if len(X_train.shape) != 3:
        raise ValueError(f"X_train must be 3D (windowed) for SSL. Got shape: {X_train.shape}")
    
    has_validation = X_val is not None and y_val is not None
    if has_validation:
        if len(X_val.shape) != 3:
             raise ValueError(f"X_val must be 3D (windowed) for validation. Got shape: {X_val.shape}")
        if len(X_val) != len(y_val):
            raise ValueError(f"X_val and y_val must have the same length. Got: {len(X_val)} and {len(y_val)}")
        X_val_win, y_val_win = X_val, y_val
        logger.info(f"Using validation data: X_val={X_val_win.shape}, y_val={len(y_val_win)}")
    
    model = EnhancedHybridEncoder(
        X_train.shape[2], args.latent_dim, dropout=args.dropout,
        transformer_seq_len=args.transformer_seq_len,
        num_transformer_layers=args.num_transformer_layers
    )
    
    # Log model architecture and parameters
    logger.info(f"Model architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {total_params} trainable parameters")
    
    if total_params == 0:
        logger.error("Model has no trainable parameters! Check architecture.")
        raise RuntimeError("Model has no trainable parameters")
    
    # Log each parameter
    for name, param in model.named_parameters():
        logger.info(f"{name}: {param.shape} (requires_grad={param.requires_grad})")
    
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler with warmup and cosine decay
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            decay_ratio = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * decay_ratio))
    
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # Stochastic Weight Averaging
    swa_model = AveragedModel(model)
    swa_start = min(args.epochs - 5, args.swa_start)
    logger.info(f"Using SWA starting at epoch {swa_start} with LR {args.lr*0.5}")
    
    # Define sensor types based on dataset
    if args.dataset == "UCI":
        # UCI: 3 body_acc, 3 body_gyro, 3 total_acc
        sensor_types = (['acc']*3 + ['gyro']*3 + ['acc']*3)
    elif args.dataset == "PAMAP2":
        # PAMAP2: 3D acc, 3D gyro, 3D mag, etc.
        sensor_types = (['acc']*3 + ['gyro']*3 + ['mag']*3 +
                       ['temp'] + ['hr'] + ['acc']*4)
    else:
        sensor_types = None
    
    # Select SSL loss
    if args.ssl_loss == "vicreg":
        ssl_loss = VICRegLoss(
            sim_weight=args.sim_weight,
            var_weight=args.var_weight,
            cov_weight=args.cov_weight,
            temp_weight=args.temp_weight
        )
        logger.info("Using enhanced VICReg loss with temporal consistency")
    else:
        ssl_loss = NTXentLoss(temperature=args.temperature)
        logger.info(f"Using NT-Xent loss with temperature={args.temperature}")
    
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    
    augmenter = SensorSpecificAugmentation(
        time_warp_limit=args.time_warp_limit,
        channel_mask_prob=args.channel_drop_prob,
        acc_noise_std=args.acc_noise_std,
        gyro_noise_std=args.gyro_noise_std,
        scale_range=(args.scale_min, args.scale_max),
        freq_mask_prob=0.2,
        sensor_types=sensor_types
    )
    logger.info(f"Using enhanced SensorSpecificAugmentation with sensor-aware noise: {sensor_types}")
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001, mode="min")
    best_model_path = f"results/best_ssl_encoder_{args.dataset.lower()}.pth"
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch in pbar:
                x = batch[0].to(device)
                with torch.no_grad():
                    x1 = augmenter(x)
                    x2 = augmenter(x)
                with autocast(enabled=use_amp):
                    z1 = model(x1)
                    z2 = model(x2)
                    loss = ssl_loss(z1, z2)
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        warmup_scheduler.step()
        
        # Update SWA after specified epoch
        if epoch >= swa_start:
            swa_model.update_parameters(model)
        
        avg_loss = total_loss / len(train_loader)
        epoch_log = f"[Epoch {epoch+1}/{args.epochs}] Loss: {avg_loss:.4f} LR: {optimizer.param_groups[0]['lr']:.6f}"
        logger.info(epoch_log)
        
        if has_validation:
            val_acc = validate_ssl_encoder(model, X_val_win, y_val_win, device)
            logger.info(f"Validation accuracy: {val_acc*100:.2f}%")
            improved = early_stopping(epoch, -val_acc)
            if improved:
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"Saved best model to {best_model_path} at epoch {epoch+1}")
            if early_stopping.should_stop:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Finalize SWA and update batch normalization with device handling
    if args.epochs > swa_start:
        logger.info("Finalizing SWA model")
        try:
            # Move model to CPU if on MPS to avoid NNPack errors
            if device.type == "mps":
                logger.info("Moving SWA model to CPU for batch norm update")
                cpu_model = swa_model.module.cpu()
                cpu_train_loader = DataLoader(
                    TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers
                )
                torch.optim.swa_utils.update_bn(cpu_train_loader, cpu_model, device=torch.device("cpu"))
                model = cpu_model.to(device)
            else:
                torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
                model = swa_model.module
        except Exception as e:
            logger.error(f"Error during SWA update: {e}")
            model = swa_model.module
            logger.info("Using SWA model without batch norm update")
    
    if has_validation and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        logger.info(f"Loaded best model from {best_model_path}")
    
    final_path = f"results/ssl_encoder_{args.dataset.lower()}.pth"
    torch.save(model.state_dict(), final_path)
    logger.info(f"Saved final SSL encoder to {final_path}")
    print_stage_footer("SSL PRETRAINING", args.dataset)
    return model

def validate_ssl_encoder(model, X_val, y_val, device):
    model.eval()
    Z_val = get_encoded_features(model, X_val, device)
    if Z_val.shape[0] != len(y_val):
        logger.error(f"Shape mismatch during validation: Z_val={Z_val.shape[0]}, y_val={len(y_val)}")
        return 0.0
    
    # Handle class imbalance
    class_counts = np.bincount(y_val)
    class_weights = {i: 1.0 / count if count > 0 else 1.0 for i, count in enumerate(class_counts)}
    
    clf = LogisticRegression(
        max_iter=2000,
        class_weight=class_weights,
        solver="saga",
        penalty="l2",
        multi_class="multinomial"
    )
    
    try:
        if len(np.unique(y_val)) < 2:
             logger.warning("Skipping validation: Less than 2 classes in validation set.")
             return 0.0
        
        # Split validation data for proxy task
        Z_val_train, Z_val_test, y_val_train, y_val_test = train_test_split(
            Z_val, y_val, test_size=0.5, stratify=y_val, random_state=42
        )
        
        # Scale features for better linear separability
        scaler = StandardScaler()
        Z_val_train = scaler.fit_transform(Z_val_train)
        Z_val_test = scaler.transform(Z_val_test)
        
        clf.fit(Z_val_train, y_val_train)
        y_pred = clf.predict(Z_val_test)
        accuracy = accuracy_score(y_val_test, y_pred)
    except Exception as e:
        logger.error(f"Error during SSL validation fit/predict: {e}")
        accuracy = 0.0
    return accuracy

def fine_tune(encoder, X_train, y_train, X_val, y_val, args):
    global subset_label_encoder
    print_stage_header("FINE TUNING", args.dataset)
    device = get_device()
    logger.info(f"Using device: {device}")
    
    if len(X_train.shape) != 3 or len(X_val.shape) != 3:
        raise ValueError("Input data must be 3D (windowed) for fine-tuning.")
    
    logger.info(f"Training data: X_train={X_train.shape}, y_train={len(y_train)}")
    logger.info(f"Validation data: X_val={X_val.shape}, y_val={len(y_val)}")
    
    if subset_label_encoder is not None:
        num_classes = len(subset_label_encoder.classes_)
        logger.info(f"Using {num_classes} classes from subset label encoder.")
    else:
        num_classes = len(np.unique(np.concatenate([y_train, y_val])))
        logger.info(f"Using {num_classes} classes found in train/val sets.")
        
    if num_classes == 0:
        raise ValueError("No classes found in training/validation data!")
        
    classifier = EnhancedClassificationHead(
        args.latent_dim,
        num_classes,
        dropout=args.dropout,
        smoothing=0.1
    ).to(device)
    
    class FullModel(nn.Module):
        def __init__(self, encoder, classifier):
            super().__init__()
            self.encoder = encoder
            self.classifier = classifier
            
        def forward(self, x, targets=None):
            z = self.encoder(x)
            if targets is not None:
                return self.classifier(z, targets)
            return self.classifier(z)
            
    model = FullModel(encoder, classifier).to(device)
    
    # Freeze encoder for first few epochs then unfreeze
    for param in encoder.parameters():
        param.requires_grad = False
        
    # Create optimizer with different learning rates
    optimizer = optim.AdamW(
        [
            {"params": classifier.parameters(), "lr": args.ft_lr}
        ],
        weight_decay=args.weight_decay
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    
    # Create class-weighted sampler
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,  # Use weighted sampler
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
    
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001, mode="max")
    best_model_path = f"results/best_finetuned_{args.dataset.lower()}.pth"
    
    # Unfreeze encoder after initial warmup
    unfreeze_after = min(5, args.ft_epochs // 2)
    
    for epoch in range(args.ft_epochs):
        # Unfreeze encoder after warmup epochs
        if epoch == unfreeze_after:
            logger.info(f"Unfreezing encoder at epoch {epoch+1}")
            for param in encoder.parameters():
                param.requires_grad = True
            optimizer.add_param_group({
                "params": encoder.parameters(),
                "lr": args.ft_lr * 0.1
            })
        
        model.train()
        train_loss, correct, total = 0, 0, 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.ft_epochs}") as pbar:
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                with autocast(enabled=use_amp):
                    loss = model(inputs, targets)
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                train_loss += loss.item()
                
                # Calculate accuracy
                with torch.no_grad():
                    logits = model(inputs)
                    _, predicted = logits.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"})
        
        train_acc = correct / total
        logger.info(f"[Epoch {epoch+1}] Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc*100:.2f}%")
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_targets, all_preds = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = F.cross_entropy(logits, targets)
                val_loss += loss.item()
                
                _, predicted = logits.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        val_acc = val_correct / val_total
        val_f1 = f1_score(all_targets, all_preds, average="weighted")
        logger.info(f"[Epoch {epoch+1}] Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc*100:.2f}%, F1: {val_f1:.4f}")
        
        scheduler.step(val_f1)
        improved = early_stopping(epoch, val_f1)
        if improved:
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model to {best_model_path} at epoch {epoch+1}")
        if early_stopping.should_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        logger.info(f"Loaded best model from {best_model_path}")
    
    final_path = f"results/finetuned_{args.dataset.lower()}.pth"
    torch.save(model.state_dict(), final_path)
    logger.info(f"Saved fine-tuned model to {final_path}")
    print_stage_footer("FINE TUNING", args.dataset)
    return model

def linear_probe(X_train, y_train, X_val, y_val, X_test, y_test, encoder, args):
    global subset_label_encoder
    print_stage_header("LINEAR PROBING", args.dataset)
    device = get_device()
    logger.info(f"Using device: {device}")
    
    if len(X_train.shape) != 3 or len(X_val.shape) != 3 or len(X_test.shape) != 3:
        raise ValueError("Input data must be 3D (windowed) for linear probe.")
    
    logger.info(f"Training data: X_train={X_train.shape}, y_train={len(y_train)}")
    logger.info(f"Validation data: X_val={X_val.shape}, y_val={len(y_val)}")
    logger.info(f"Test data: X_test={X_test.shape}, y_test={len(y_test)}")
    
    if subset_label_encoder is not None:
        all_original_labels = subset_label_encoder.classes_
        num_classes_total = len(all_original_labels)
        logger.info(f"Using {num_classes_total} re-encoded classes for subset: {all_original_labels}")
    else:
        all_labels_combined = np.concatenate([y_train, y_val, y_test])
        all_original_labels = np.unique(all_labels_combined)
        num_classes_total = len(all_original_labels)
        logger.info(f"Using {num_classes_total} original classes: {all_original_labels}")
        
    logger.info(f"Train class distribution (encoded): {np.bincount(y_train, minlength=num_classes_total)}")
    logger.info(f"Val class distribution (encoded): {np.bincount(y_val, minlength=num_classes_total)}")
    logger.info(f"Test class distribution (encoded): {np.bincount(y_test, minlength=num_classes_total)}")
    
    encoder.eval()
    logger.info("Extracting features...")
    
    # Extract features
    Z_train = get_encoded_features(model=encoder, X=X_train, device=device, batch_size=args.batch_size)
    Z_val = get_encoded_features(model=encoder, X=X_val, device=device, batch_size=args.batch_size)
    Z_test = get_encoded_features(model=encoder, X=X_test, device=device, batch_size=args.batch_size)
    
    # Standardize features
    logger.info("Standardizing features for linear probe...")
    scaler = StandardScaler()
    Z_train_scaled = scaler.fit_transform(Z_train)
    Z_val_scaled = scaler.transform(Z_val)
    Z_test_scaled = scaler.transform(Z_test)
    
    classifiers = []
    classifier_names = []
    
    # Class weights for imbalanced data
    class_counts = np.bincount(y_train)
    class_weights = {i: 1.0 / count if count > 0 else 1.0 for i, count in enumerate(class_counts)}
    
    # Logistic Regression
    logger.info("Training Logistic Regression classifier...")
    lr_params = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["saga"],
        "penalty": ["l2"],
        "class_weight": [class_weights, None],
        "max_iter": [5000]
    }
    lr_clf = GridSearchCV(
        LogisticRegression(multi_class="multinomial"),
        lr_params,
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1
    )
    lr_clf.fit(Z_train_scaled, y_train)
    logger.info(f"Best LogisticRegression params: {lr_clf.best_params_}")
    classifiers.append(lr_clf.best_estimator_)
    classifier_names.append("LogisticRegression")
    
    # Support Vector Machine
    logger.info("Training SVM classifier...")
    svm_params = {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
        "class_weight": ["balanced", None]
    }
    svm_clf = GridSearchCV(
        SVC(probability=True),
        svm_params,
        cv=2,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1
    )
    svm_clf.fit(Z_train_scaled, y_train)
    logger.info(f"Best SVM params: {svm_clf.best_params_}")
    classifiers.append(svm_clf.best_estimator_)
    classifier_names.append("SVM")
    
    # Random Forest
    logger.info("Training Random Forest classifier...")
    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "class_weight": ["balanced", None]
    }
    rf_clf = GridSearchCV(
        RandomForestClassifier(),
        rf_params,
        cv=2,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1
    )
    rf_clf.fit(Z_train_scaled, y_train)
    logger.info(f"Best RandomForest params: {rf_clf.best_params_}")
    classifiers.append(rf_clf.best_estimator_)
    classifier_names.append("RandomForest")
    
    val_scores = []
    for i, clf in enumerate(classifiers):
        y_pred_val = clf.predict(Z_val_scaled)
        val_acc = accuracy_score(y_val, y_pred_val)
        val_f1 = f1_score(y_val, y_pred_val, average="weighted")
        val_scores.append((val_acc, val_f1))
        logger.info(f"{classifier_names[i]} - Val Accuracy: {val_acc*100:.2f}%, F1: {val_f1:.4f}")
        
    weights = [score[1] for score in val_scores]
    weights = np.array(weights) / sum(weights) if sum(weights) > 0 else np.ones(len(classifiers)) / len(classifiers)
    ensemble = EnsemblePredictor(classifiers, weights)
    best_idx = np.argmax([score[1] for score in val_scores])
    best_clf = classifiers[best_idx]
    best_clf_name = classifier_names[best_idx]
    
    # Create stacked classifier
    logger.info("Training Stacked Classifier...")
    base_estimators = [
        ('lr', lr_clf.best_estimator_),
        ('svm', svm_clf.best_estimator_),
        ('rf', rf_clf.best_estimator_)
    ]
    
    stacked_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            multi_class='multinomial'
        ),
        cv=5,
        n_jobs=args.num_workers
    )
    
    # Train on combined train+val data
    Z_combined = np.vstack((Z_train_scaled, Z_val_scaled))
    y_combined = np.concatenate((y_train, y_val))
    stacked_clf.fit(Z_combined, y_combined)
    
    joblib.dump(best_clf, f"results/best_classifier_{args.dataset.lower()}.pkl")
    joblib.dump(ensemble, f"results/ensemble_classifier_{args.dataset.lower()}.pkl")
    joblib.dump(stacked_clf, f"results/stacked_classifier_{args.dataset.lower()}.pkl")
    logger.info(f"Saved best classifier ({best_clf_name}), ensemble and stacked classifier to results/")
    
    results_filename = f"results/{args.dataset.lower()}_linear_probe_metrics.txt"
    all_metrics = {}
    
    def calculate_and_store_metrics(y_true, y_pred, name, proba=None):
        if y_pred is None or not isinstance(y_pred, np.ndarray) or y_pred.ndim == 0 or len(y_pred) != len(y_true):
            logger.warning(f"Skipping metrics calculation for {name}: Invalid predictions")
            return {f"{name}_Error": "Invalid predictions"}
            
        try:
            present_labels_encoded = np.unique(np.concatenate([y_true, y_pred]))
        except ValueError as e:
            logger.error(f"Error concatenating labels: {e}")
            return {f"{name}_Error": "Incompatible prediction dimensions"}
            
        if len(present_labels_encoded) == 0:
            logger.warning(f"Skipping metrics calculation for {name}: No labels found")
            return {f"{name}_Error": "No labels found"}
            
        if subset_label_encoder is not None:
            try:
                present_labels_original = subset_label_encoder.inverse_transform(present_labels_encoded)
                present_target_names = [str(c) for c in present_labels_original]
            except Exception as e:
                logger.error(f"Error inverse transforming labels: {e}")
                present_labels_original = present_labels_encoded
                present_target_names = [str(c) for c in present_labels_encoded]
        else:
            present_labels_original = present_labels_encoded
            present_target_names = [str(c) for c in present_labels_original]
            
        logger.info(f"Calculating metrics for {name} using {len(present_labels_original)} classes")

        try:
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, labels=present_labels_encoded, average="weighted", zero_division=0)
            prec = precision_score(y_true, y_pred, labels=present_labels_encoded, average="weighted", zero_division=0)
            rec = recall_score(y_true, y_pred, labels=present_labels_encoded, average="weighted", zero_division=0)
            report = classification_report(y_true, y_pred, labels=present_labels_encoded,
                                          target_names=present_target_names, zero_division=0)
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {f"{name}_Error": f"Metrics calculation failed: {e}"}
            
        metrics_dict = {
            f"{name}_Accuracy": f"{acc*100:.2f}%",
            f"{name}_F1_Score": f"{f1:.4f}",
            f"{name}_Precision": f"{prec:.4f}",
            f"{name}_Recall": f"{rec:.4f}",
            f"{name}_Classification_Report": "\n" + report
        }
        print(f"\n📊 {name} METRICS:")
        print(f"| Accuracy: {acc*100:.2f}%\n| F1 Score: {f1:.4f}\n| Precision: {prec:.4f}\n| Recall: {rec:.4f}")
        print("\n📝 Classification Report:")
        print(report)
        logger.info(f"{name} METRICS: Acc={acc*100:.2f}%, F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")
        return metrics_dict
        
    # Evaluate best classifier
    y_pred_best = best_clf.predict(Z_test_scaled)
    y_proba_best = best_clf.predict_proba(Z_test_scaled) if hasattr(best_clf, "predict_proba") else None
    best_metrics = calculate_and_store_metrics(y_test, y_pred_best, f"BestClassifier_{best_clf_name}", y_proba_best)
    all_metrics.update(best_metrics)
    
    # Evaluate ensemble
    try:
        y_pred_ensemble = ensemble.predict(Z_test_scaled)
        y_proba_ensemble = ensemble.predict_proba(Z_test_scaled) if hasattr(ensemble, "predict_proba") else None
        ensemble_metrics = calculate_and_store_metrics(y_test, y_pred_ensemble, "Ensemble", y_proba_ensemble)
        all_metrics.update(ensemble_metrics)
    except Exception as e:
        logger.error(f"Error getting predictions from ensemble: {e}")
        all_metrics["Ensemble_Error"] = str(e)
    
    # Evaluate stacked classifier
    try:
        y_pred_stacked = stacked_clf.predict(Z_test_scaled)
        y_proba_stacked = stacked_clf.predict_proba(Z_test_scaled)
        stacked_metrics = calculate_and_store_metrics(y_test, y_pred_stacked, "StackedClassifier", y_proba_stacked)
        all_metrics.update(stacked_metrics)
    except Exception as e:
        logger.error(f"Error getting predictions from stacked classifier: {e}")
        all_metrics["StackedClassifier_Error"] = str(e)
    
    save_metrics_to_file(all_metrics, results_filename)
    
    # Visualization
    visualize_results(y_test, y_pred_stacked, Z_test_scaled, best_clf, args)
    
    print_stage_footer("LINEAR PROBING", args.dataset)
    return stacked_clf

def visualize_results(y_test, y_pred, Z_test, best_clf, args):
    """Generate visualizations for results"""
    if subset_label_encoder is not None:
        binarize_classes = subset_label_encoder.classes_
    else:
        binarize_classes = np.unique(y_test)
    
    # Confusion Matrix
    if y_pred is not None and len(y_pred) == len(y_test):
        present_labels = np.unique(np.concatenate([y_test, y_pred]))
        if subset_label_encoder is not None:
            try:
                present_labels_original = subset_label_encoder.inverse_transform(present_labels)
            except Exception:
                present_labels_original = present_labels
        else:
            present_labels_original = present_labels
            
        num_classes = len(present_labels)
        cm_filename = f"results/{args.dataset.lower()}_confusion_matrix_ensemble.png"
        try:
            cm = confusion_matrix(y_test, y_pred, labels=present_labels)
            plt.figure(figsize=(max(8, num_classes), max(6, num_classes)))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=present_labels_original,
                        yticklabels=present_labels_original)
            plt.title(f"Confusion Matrix - {args.dataset}", fontsize=12)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.savefig(cm_filename, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved confusion matrix to {cm_filename}")
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
    
    # ROC Curve
    if hasattr(best_clf, "predict_proba") and len(binarize_classes) > 1:
        roc_filename = f"results/{args.dataset.lower()}_roc_curves_best_clf.png"
        try:
            y_test_bin = label_binarize(y_test, classes=binarize_classes)
            y_proba = best_clf.predict_proba(Z_test)
            
            # Align probabilities with binarized classes
            aligned_proba = np.zeros((y_proba.shape[0], len(binarize_classes)))
            clf_classes = best_clf.classes_
            for i, cls_orig in enumerate(binarize_classes):
                try:
                    clf_idx = np.where(clf_classes == cls_orig)[0][0]
                    aligned_proba[:, i] = y_proba[:, clf_idx]
                except Exception:
                    aligned_proba[:, i] = 0
            
            # Compute ROC curves
            fpr, tpr, roc_auc = {}, {}, {}
            plt.figure(figsize=(8, 6))
            for i in range(len(binarize_classes)):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], aligned_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i], lw=1.5,
                         label=f"Class {binarize_classes[i]} (AUC={roc_auc[i]:.2f})")
            
            plt.plot([0, 1], [0, 1], "k--", lw=1.5)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curves - {args.dataset}", fontsize=12)
            plt.legend(loc="lower right", fontsize="small")
            plt.savefig(roc_filename, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved ROC curves to {roc_filename}")
        except Exception as e:
            logger.error(f"Error plotting ROC curves: {e}")
    
    # t-SNE Visualization
    tsne_filename = f"results/{args.dataset.lower()}_tsne_embeddings.png"
    if Z_test.shape[0] > 1 and Z_test.shape[1] > 1:
        try:
            # Use PCA first for dimensionality reduction
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(50, Z_test.shape[1]))
            Z_pca = pca.fit_transform(Z_test)
            
            tsne = TSNE(n_components=2, perplexity=min(30, Z_pca.shape[0]-1),
                         n_iter=1000, random_state=42, method="barnes_hut")
            Z_2d = tsne.fit_transform(Z_pca)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(Z_2d[:, 0], Z_2d[:, 1], c=y_test, cmap="viridis", s=15, alpha=0.7)
            
            if subset_label_encoder is not None:
                try:
                    present_labels = np.unique(y_test)
                    present_labels_original = subset_label_encoder.inverse_transform(present_labels)
                except Exception:
                    present_labels_original = present_labels
            else:
                present_labels_original = np.unique(y_test)
                
            plt.colorbar(scatter, ticks=np.unique(y_test))
            plt.title(f"t-SNE Embeddings - {args.dataset}", fontsize=12)
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.savefig(tsne_filename, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved t-SNE plot to {tsne_filename}")
        except Exception as e:
            logger.error(f"Error plotting t-SNE: {e}")

def evaluate_alignment_with_mh(X, y, user_ids, encoder, args):
    global subset_label_encoder
    print_stage_header("ALIGNMENT EVALUATION", args.dataset)
    device = get_device()
    logger.info(f"Using device: {device}")
    
    if len(X.shape) != 3:
        raise ValueError(f"X must be 3D (windowed) for alignment evaluation. Got shape: {X.shape}")
    
    logger.info(f"Data: X={X.shape}, y={len(y)}, user_ids={len(user_ids)}")
    encoder.eval()
    logger.info("Extracting features...")
    Z = get_encoded_features(model=encoder, X=X, device=device, batch_size=args.batch_size)
    
    unique_users = np.unique(user_ids)
    if len(unique_users) < 2:
        logger.warning("Skipping alignment evaluation: Need at least 2 users.")
        print_stage_footer("ALIGNMENT EVALUATION", args.dataset)
        return 0, 0, 0
    
    n_splits = min(5, len(unique_users) // 2)
    if n_splits == 0:
        logger.warning("Skipping alignment evaluation: Not enough users for splits.")
        print_stage_footer("ALIGNMENT EVALUATION", args.dataset)
        return 0, 0, 0
    
    # Add downsampling for large datasets
    max_windows_per_user = 100 if args.dataset == "UCI" else 200
    max_total_windows = 1500
    
    accuracies, f1_scores, temporal_coherences = [], [], []
    
    for split in range(n_splits):
        np.random.shuffle(unique_users)
        split_idx = len(unique_users) // 2
        source_users, target_users = unique_users[:split_idx], unique_users[split_idx:]
        
        try:
            # Downsample windows per user
            def get_sampled_indices(users):
                indices = []
                for user in users:
                    user_idx = np.where(user_ids == user)[0]
                    if len(user_idx) > max_windows_per_user:
                        user_idx = np.random.choice(
                            user_idx,
                            size=max_windows_per_user,
                            replace=False
                        )
                    indices.extend(user_idx)
                return np.array(indices)
            
            source_indices = get_sampled_indices(source_users)
            target_indices = get_sampled_indices(target_users)
            
            # Further downsample if still too large
            if len(source_indices) > max_total_windows:
                source_indices = np.random.choice(source_indices, max_total_windows, replace=False)
            if len(target_indices) > max_total_windows:
                target_indices = np.random.choice(target_indices, max_total_windows, replace=False)
                
            Z_source = Z[source_indices]
            Z_target = Z[target_indices]
            y_source = y[source_indices]
            y_target = y[target_indices]
            
            logger.info(f"Split {split+1}/{n_splits}: Using {len(Z_source)} source and {len(Z_target)} target windows")
            
            # Matrix chunking for large datasets
            def chunked_hungarian(aligner, Z1, Z2):
                if len(Z1) * len(Z2) > 1e7:  # 10M element threshold
                    chunk_size = 500
                    assignments = []
                    for i in range(0, len(Z1), chunk_size):
                        for j in range(0, len(Z2), chunk_size):
                            Z1_chunk = Z1[i:i+chunk_size]
                            Z2_chunk = Z2[j:j+chunk_size]
                            row_ind, col_ind = aligner(Z1_chunk, Z2_chunk)
                            assignments.extend([(i + ri, j + ci) for ri, ci in zip(row_ind, col_ind)])
                    return zip(*assignments)
                else:
                    return aligner(Z1, Z2)
            
            # Use both alignment methods
            for aligner_name, aligner in [
                ("QuantumHungarian", QuantumHungarian(temp=ALIGNMENT_TEMP, n_iter=ALIGNMENT_N_ITER, feature_weight=args.feature_weight)),
                ("SinkhornAlign", SinkhornAlign(n_iter=20, epsilon=0.1))
            ]:
                try:
                    if "Quantum" in aligner_name and len(Z_source) * len(Z_target) > 2e6:
                        logger.info(f"Using chunked Hungarian for large matrix: {len(Z_source)}x{len(Z_target)}")
                        row_ind, col_ind = chunked_hungarian(aligner, Z_source, Z_target)
                    else:
                        row_ind, col_ind = aligner(Z_source, Z_target)
                    
                    # Convert labels to original space if needed
                    if subset_label_encoder:
                        y_source_orig = subset_label_encoder.inverse_transform(y_source)
                        y_target_orig = subset_label_encoder.inverse_transform(y_target)
                    else:
                        y_source_orig, y_target_orig = y_source, y_target
                    
                    acc, f1, temporal = evaluate_alignment(y_source_orig, y_target_orig, row_ind, col_ind)
                    logger.info(f"Split {split+1}/{n_splits}, {aligner_name} - Acc: {acc*100:.2f}%, F1: {f1:.4f}, Temporal: {temporal:.4f}")
                    
                    # Store best result for this split
                    if aligner_name == "QuantumHungarian" or (f1_scores and f1 > max(f1_scores)):
                        if f1_scores and aligner_name != "QuantumHungarian":
                            # Replace previous result if this is better
                            accuracies[-1] = acc
                            f1_scores[-1] = f1
                            temporal_coherences[-1] = temporal
                        else:
                            accuracies.append(acc)
                            f1_scores.append(f1)
                            temporal_coherences.append(temporal)
                except Exception as e:
                    logger.error(f"Error during {aligner_name} alignment in split {split+1}: {e}")
                    logger.exception(e)
        
        except Exception as e:
            logger.error(f"Critical error in split {split+1}: {e}")
            logger.exception(e)
            continue
    
    avg_acc = np.mean(accuracies) if accuracies else 0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0
    avg_temporal = np.mean(temporal_coherences) if temporal_coherences else 0
    
    results_msg = (f"\n🎯 Cross-User Alignment Results (Avg over {len(accuracies)} valid splits):\n"
                  f"Accuracy: {avg_acc*100:.2f}%\n"
                  f"F1 Score: {avg_f1:.4f}\n"
                  f"Temporal Coherence: {avg_temporal:.4f}")
    print(results_msg)
    logger.info(results_msg)
    
    alignment_metrics = {
        "Average_Accuracy": f"{avg_acc*100:.2f}%",
        "Average_F1_Score": f"{avg_f1:.4f}",
        "Average_Temporal_Coherence": f"{avg_temporal:.4f}",
        "Number_of_Valid_Splits": len(accuracies)
    }
    results_filename = f"results/{args.dataset.lower()}_alignment_metrics.txt"
    save_metrics_to_file(alignment_metrics, results_filename)
    
    print_stage_footer("ALIGNMENT EVALUATION", args.dataset)
    return avg_acc, avg_f1, avg_temporal

# ---------------------- Main Function ----------------------
def main():
    global subset_label_encoder
    parser = argparse.ArgumentParser(description="Enhanced HAR SSL Pipeline (v8.1)")
    parser.add_argument("--mode", type=str, default="all", choices=["pretrain", "finetune", "linear_probe", "alignment", "all"], help="Pipeline mode")
    parser.add_argument("--dataset", type=str, default="PAMAP2", choices=["UCI", "PAMAP2"], help="Dataset to use")
    parser.add_argument("--latent_dim", type=int, default=256, help="Latent dimension size")
    parser.add_argument("--transformer_seq_len", type=int, default=32, help="Sequence length for transformer input")
    parser.add_argument("--num_transformer_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--window_size", type=int, default=192, help="Window size")
    parser.add_argument("--window_step", type=int, default=64, help="Step size")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs for SSL pretraining")
    parser.add_argument("--ft_epochs", type=int, default=30, help="Epochs for fine-tuning")
    parser.add_argument("--lr", type=float, default=0.0005, help="LR for SSL pretraining")
    parser.add_argument("--ft_lr", type=float, default=0.001, help="LR for fine-tuning")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping")
    parser.add_argument("--num_workers", type=int, default=2, help="Data loading workers")
    parser.add_argument("--ssl_loss", type=str, default="ntxent", choices=["ntxent", "vicreg"], help="SSL loss function")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temp for contrastive loss")
    parser.add_argument("--sim_weight", type=float, default=25.0, help="Sim weight for VICReg")
    parser.add_argument("--var_weight", type=float, default=25.0, help="Var weight for VICReg")
    parser.add_argument("--cov_weight", type=float, default=1.0, help="Cov weight for VICReg")
    parser.add_argument("--time_warp_limit", type=float, default=0.2, help="Time warping limit")
    parser.add_argument("--channel_drop_prob", type=float, default=0.3, help="Channel dropout probability")
    parser.add_argument("--acc_noise_std", type=float, default=0.1, help="Accelerometer noise std")
    parser.add_argument("--gyro_noise_std", type=float, default=0.15, help="Gyroscope noise std")
    parser.add_argument("--scale_min", type=float, default=0.8, help="Minimum scaling factor")
    parser.add_argument("--scale_max", type=float, default=1.2, help="Maximum scaling factor")
    parser.add_argument("--feature_weight", type=float, default=0.7, help="Feature weight for QuantumHungarian")
    parser.add_argument("--subset_fraction", type=float, default=1.0, help="Fraction of data to use")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Warm-up epochs for LR")
    parser.add_argument("--swa_start", type=int, default=70, help="Epoch to start SWA")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--temp_weight", type=float, default=0.3, help="Temporal consistency loss weight")
    
    args = parser.parse_args()
    logger.info(f"Starting pipeline (v8.1) with arguments: {vars(args)}")
    os.makedirs("results", exist_ok=True)
    logger.info("Created/verified 'results' directory for outputs.")
    
    logger.info(f"Loading {args.dataset} dataset...")
    try:
        loaded_data = load_data(args.dataset)
        
        if args.dataset == "UCI":
            X_train_raw, y_train_raw, X_test_raw, y_test_raw, user_train_raw, user_test_raw = loaded_data
            X_train_raw, X_val_raw, y_train_raw, y_val_raw, user_train_raw, user_val_raw = train_test_split(
                X_train_raw, y_train_raw, user_train_raw, test_size=0.2, stratify=y_train_raw, random_state=42
            )
            n_channels = 9
            seq_len = 128
            # Reshape to [samples, seq_len, channels]
            X_train_raw = X_train_raw.reshape(-1, seq_len, n_channels)
            X_val_raw = X_val_raw.reshape(-1, seq_len, n_channels)
            X_test_raw = X_test_raw.reshape(-1, seq_len, n_channels)
            X_train, y_train, user_train = X_train_raw, y_train_raw, user_train_raw
            X_val, y_val, user_val = X_val_raw, y_val_raw, user_val_raw
            X_test, y_test, user_test = X_test_raw, y_test_raw, user_test_raw
            logger.info(f"UCI data reshaped: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
        elif args.dataset == "PAMAP2":
            X_train, y_train, X_val, y_val, X_test, y_test, user_train, user_val, user_test = loaded_data
            logger.info(f"PAMAP2 data loaded: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
            
    except Exception as e:
        logger.error(f"Failed to load or process dataset '{args.dataset}': {e}")
        raise
        
    if args.subset_fraction < 1.0:
        logger.warning(f"Using only {args.subset_fraction*100:.1f}% of the data for lightweight testing.")
        def take_subset(X, y, users, fraction):
            num_samples = int(len(X) * fraction)
            if num_samples == 0: return X[:1], y[:1], users[:1]
            indices = np.random.choice(len(X), num_samples, replace=False)
            return X[indices], y[indices], users[indices]
        
        X_train, y_train, user_train = take_subset(X_train, y_train, user_train, args.subset_fraction)
        X_val, y_val, user_val = take_subset(X_val, y_val, user_val, args.subset_fraction)
        X_test, y_test, user_test = take_subset(X_test, y_test, user_test, args.subset_fraction)
        logger.info(f"Subset sizes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

        all_subset_labels = np.concatenate([y_train, y_val, y_test])
        unique_subset_labels = np.unique(all_subset_labels)
        
        if len(unique_subset_labels) == 0:
            raise ValueError("No labels found in the data subset!")
            
        subset_label_encoder = LabelEncoder()
        subset_label_encoder.fit(unique_subset_labels)
        
        y_train = subset_label_encoder.transform(y_train)
        y_val = subset_label_encoder.transform(y_val)
        y_test = subset_label_encoder.transform(y_test)
        
        num_classes_subset = len(subset_label_encoder.classes_)
        logger.info(f"Re-encoded labels for subset. Original unique labels in subset: {unique_subset_labels}. New number of classes: {num_classes_subset}")
    else:
        subset_label_encoder = None

    encoder = None
    if args.mode == "pretrain" or args.mode == "all":
        encoder = train_ssl(X_train, X_val, y_val, args)
    
    if encoder is None and (args.mode != "pretrain" or args.mode == "all"):
        encoder_path = f"results/ssl_encoder_{args.dataset.lower()}.pth"
        if not os.path.exists(encoder_path):
            encoder_path = f"results/best_ssl_encoder_{args.dataset.lower()}.pth"
        
        if os.path.exists(encoder_path):
            logger.info(f"Loading pre-trained encoder from {encoder_path}")
            input_dim = X_train.shape[2]
            encoder = EnhancedHybridEncoder(
                input_dim, args.latent_dim, dropout=args.dropout,
                transformer_seq_len=args.transformer_seq_len,
                num_transformer_layers=args.num_transformer_layers
            )
            try:
                encoder.load_state_dict(torch.load(encoder_path, map_location=get_device()))
            except Exception as e:
                logger.error(f"Failed to load encoder state dict: {e}")
                raise
            encoder = encoder.to(get_device())
        elif args.mode != "pretrain":
            logger.error(f"Pre-trained encoder not found at {encoder_path}")
            raise FileNotFoundError(f"Encoder not found: {encoder_path}")
    
    if args.mode == "finetune" or args.mode == "all":
        fine_tuned_model = fine_tune(encoder, X_train, y_train, X_val, y_val, args)
    
    if args.mode == "linear_probe" or args.mode == "all":
        ensemble = linear_probe(X_train, y_train, X_val, y_val, X_test, y_test, encoder, args)
    
    if args.mode == "alignment" or args.mode == "all":
        X_all = np.concatenate([X_train, X_val, X_test])
        y_all = np.concatenate([y_train, y_val, y_test])
        user_all = np.concatenate([user_train, user_val, user_test])
        evaluate_alignment_with_mh(X_all, y_all, user_all, encoder, args)
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    logger.info(f"Total execution time: {elapsed_time/60:.2f} minutes")
