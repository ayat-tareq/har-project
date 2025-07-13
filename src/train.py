#!/usr/bin/env python3
"""
Human Activity Recognition (HAR) Pipeline with SSL Pre-training and Evaluation

Version 6 - Lightweight v4: Integrates strategies from train1.py, fixes errors,
                            organizes result saving, reduces resource usage,
                            fixes label encoding after subset sampling,
                            dynamically handles labels/target_names in reports/plots,
                            and adds checks for empty/invalid predictions in metrics.
                            - Reduced latent_dim, transformer_seq_len, num_layers
                            - Reduced default batch_size, epochs, ft_epochs, num_workers
                            - Added label re-encoding after subset sampling
                            - Dynamic labels/target_names for metrics and plots
                            - Added checks for empty/invalid y_pred/proba in metrics
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
from sklearn.preprocessing import label_binarize, LabelEncoder
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

# --- Constants for fixed parameters (inspired by train1.py) ---
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

# ---------------------- Model Components (Lightweight) ----------------------
class PositionalEncoding(nn.Module):
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

# Lightweight Simplified Encoder
class SimplifiedHybridEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128, dropout=0.3, transformer_seq_len=32, num_transformer_layers=1):
        super().__init__()
        self.transformer_seq_len = transformer_seq_len
        self.conv_branch = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(self.transformer_seq_len)
        )
        self.pos_enc = PositionalEncoding(d_model=128, dropout=dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128, nhead=4, dim_feedforward=512,
                dropout=dropout, batch_first=True
            ),
            num_layers=num_transformer_layers
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * self.transformer_seq_len, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_branch(x)
        x = x.permute(0, 2, 1)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.projector(x)

class ClassificationHead(nn.Module):
    def __init__(self, latent_dim, num_classes, dropout=0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

# Simplified Augmentation (from train1.py)
class SensorAugmentation:
    def __init__(self, time_warp_limit=0.2, channel_drop_prob=0.2, noise_std=0.1, scale_range=(0.8, 1.2)):
        self.time_warp_limit = time_warp_limit
        self.channel_drop_prob = channel_drop_prob
        self.noise_std = noise_std
        self.scale_range = scale_range
        
    def __call__(self, x):
        x = self.time_warp(x)
        if self.channel_drop_prob > 0:
            mask = torch.ones_like(x)
            for c in range(x.shape[1]):
                if random.random() < self.channel_drop_prob:
                    mask[:, c, :] = 0
            x = x * mask
        scale_factors = torch.FloatTensor(x.shape[1]).uniform_(*self.scale_range)
        x = x * scale_factors[None, :, None].to(x.device)
        noise = torch.normal(0, self.noise_std, size=x.shape, device=x.device)
        x = x + noise
        return x
    
    def time_warp(self, x):
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

class VICRegLoss(nn.Module):
    def __init__(self, sim_weight=25.0, var_weight=25.0, cov_weight=1.0):
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        
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
        loss = self.sim_weight * sim_loss + self.var_weight * var_loss + self.cov_weight * cov_loss
        return loss

class ContrastiveLoss(nn.Module):
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
    
    model = SimplifiedHybridEncoder(
        X_train.shape[2], args.latent_dim, dropout=args.dropout,
        transformer_seq_len=args.transformer_seq_len,
        num_transformer_layers=args.num_transformer_layers
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    if args.ssl_loss == "vicreg":
        ssl_loss = VICRegLoss(sim_weight=args.sim_weight, var_weight=args.var_weight, cov_weight=args.cov_weight)
        logger.info("Using VICReg loss")
    else:
        ssl_loss = ContrastiveLoss(temperature=args.temperature)
        logger.info(f"Using Contrastive loss with temperature={args.temperature}")
    
    use_amp = device.type in ["cuda", "mps"]
    scaler = GradScaler(enabled=use_amp)
    augmenter = SensorAugmentation(
        time_warp_limit=args.time_warp_limit, channel_drop_prob=args.channel_drop_prob,
        noise_std=args.noise_std, scale_range=(args.scale_min, args.scale_max)
    )
    logger.info("Using simplified SensorAugmentation")
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    early_stopping = EarlyStopping(patience=args.patience, mode="min")
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
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        avg_loss = total_loss / len(train_loader)
        epoch_log = f"[Epoch {epoch+1}/{args.epochs}] Loss: {avg_loss:.4f}"
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
        else:
            if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                torch.save(model.state_dict(), f"results/ssl_encoder_{args.dataset.lower()}_epoch{epoch+1}.pth")
        scheduler.step()
    
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
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    try:
        # Ensure y_val has valid labels for the classifier
        if len(np.unique(y_val)) < 2:
             logger.warning("Skipping validation: Less than 2 classes in validation set.")
             return 0.0
        clf.fit(Z_val, y_val)
        y_pred = clf.predict(Z_val)
        accuracy = accuracy_score(y_val, y_pred)
    except Exception as e:
        logger.error(f"Error during SSL validation fit/predict: {e}")
        accuracy = 0.0
    return accuracy

def fine_tune(encoder, X_train, y_train, X_val, y_val, args):
    global subset_label_encoder # Access the global encoder
    print_stage_header("FINE TUNING", args.dataset)
    device = get_device()
    logger.info(f"Using device: {device}")
    if len(X_train.shape) != 3 or len(X_val.shape) != 3:
        raise ValueError("Input data must be 3D (windowed) for fine-tuning.")
    logger.info(f"Training data: X_train={X_train.shape}, y_train={len(y_train)}")
    logger.info(f"Validation data: X_val={X_val.shape}, y_val={len(y_val)}")
    
    # Determine number of classes based on whether subset was used and re-encoded
    if subset_label_encoder is not None:
        num_classes = len(subset_label_encoder.classes_)
        logger.info(f"Using {num_classes} classes from subset label encoder.")
    else:
        num_classes = len(np.unique(np.concatenate([y_train, y_val])))
        logger.info(f"Using {num_classes} classes found in train/val sets.")
        
    if num_classes == 0:
        raise ValueError("No classes found in training/validation data!")
        
    classifier = ClassificationHead(args.latent_dim, num_classes, dropout=args.dropout).to(device)
    class FullModel(nn.Module):
        def __init__(self, encoder, classifier):
            super().__init__()
            self.encoder = encoder
            self.classifier = classifier
        def forward(self, x):
            return self.classifier(self.encoder(x))
    model = FullModel(encoder, classifier).to(device)
    optimizer = optim.AdamW([
        {"params": encoder.parameters(), "lr": args.ft_lr * 0.1},
        {"params": classifier.parameters(), "lr": args.ft_lr}
    ], weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    use_amp = device.type in ["cuda", "mps"]
    scaler = GradScaler(enabled=use_amp)
    early_stopping = EarlyStopping(patience=args.patience, mode="max")
    best_model_path = f"results/best_finetuned_{args.dataset.lower()}.pth"
    for epoch in range(args.ft_epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.ft_epochs}") as pbar:
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                with autocast(enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"})
        train_acc = correct / total
        logger.info(f"[Epoch {epoch+1}] Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc*100:.2f}%")
        model.eval()
        val_loss, correct, total = 0, 0, 0
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
        scheduler.step(val_acc)
        improved = early_stopping(epoch, val_acc)
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
    global subset_label_encoder # Access the global encoder
    print_stage_header("LINEAR PROBING", args.dataset)
    device = get_device()
    logger.info(f"Using device: {device}")
    if len(X_train.shape)!= 3 or len(X_val.shape)!= 3 or len(X_test.shape)!= 3:
        raise ValueError("Input data must be 3D (windowed) for linear probe.")
    logger.info(f"Training data: X_train={X_train.shape}, y_train={len(y_train)}")
    logger.info(f"Validation data: X_val={X_val.shape}, y_val={len(y_val)}")
    logger.info(f"Test data: X_test={X_test.shape}, y_test={len(y_test)}")
    
    # Determine classes and labels based on whether subset was used
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
    Z_train = get_encoded_features(model=encoder, X=X_train, device=device, batch_size=args.batch_size)
    Z_val = get_encoded_features(model=encoder, X=X_val, device=device, batch_size=args.batch_size)
    Z_test = get_encoded_features(model=encoder, X=X_test, device=device, batch_size=args.batch_size)
    
    classifiers = []
    classifier_names = []
    logger.info("Training Logistic Regression classifier...")
    lr_params = {"C": [0.1, 1.0], "solver": ["liblinear"], "penalty": ["l2"]}
    lr_clf = GridSearchCV(
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        lr_params, cv=2, scoring="f1_weighted", n_jobs=-1
    )
    lr_clf.fit(Z_train, y_train)
    logger.info(f"Best LogisticRegression params: {lr_clf.best_params_}")
    classifiers.append(lr_clf.best_estimator_)
    classifier_names.append("LogisticRegression")
    
    val_scores = []
    for i, clf in enumerate(classifiers):
        y_pred_val = clf.predict(Z_val)
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
    joblib.dump(best_clf, f"results/best_classifier_{args.dataset.lower()}.pkl")
    joblib.dump(ensemble, f"results/ensemble_classifier_{args.dataset.lower()}.pkl")
    logger.info(f"Saved best classifier ({best_clf_name}) and ensemble to results/")
    
    results_filename = f"results/{args.dataset.lower()}_linear_probe_metrics.txt"
    all_metrics = {}
    
    # --- Dynamic Label Handling for Metrics (with checks) ---
    def calculate_and_store_metrics(y_true, y_pred, name, proba=None):
        # Check if y_pred is valid
        if y_pred is None or not isinstance(y_pred, np.ndarray) or y_pred.ndim == 0 or len(y_pred) != len(y_true):
            logger.warning(f"Skipping metrics calculation for {name}: Invalid or incompatible y_pred (shape: {y_pred.shape if isinstance(y_pred, np.ndarray) else type(y_pred)}, expected: ({len(y_true)},))")
            return {f"{name}_Error": "Invalid predictions"}
            
        # Determine labels present in this specific y_true/y_pred pair
        try:
            present_labels_encoded = np.unique(np.concatenate([y_true, y_pred]))
        except ValueError as e:
            logger.error(f"Error concatenating y_true and y_pred for {name}: {e}. y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
            return {f"{name}_Error": "Incompatible prediction dimensions"}
            
        if len(present_labels_encoded) == 0:
            logger.warning(f"Skipping metrics calculation for {name}: No labels found in y_true/y_pred.")
            return {f"{name}_Error": "No labels found"}
            
        # Get original names for the present labels
        if subset_label_encoder is not None:
            try:
                present_labels_original = subset_label_encoder.inverse_transform(present_labels_encoded)
                present_target_names = [str(c) for c in present_labels_original]
            except Exception as e:
                logger.error(f"Error inverse transforming labels for {name}: {e}. Using encoded labels as names.")
                present_labels_original = present_labels_encoded
                present_target_names = [str(c) for c in present_labels_encoded]
        else:
            present_labels_original = present_labels_encoded
            present_target_names = [str(c) for c in present_labels_original]
            
        logger.info(f"Calculating metrics for {name} using labels: {present_labels_encoded} (Original: {present_labels_original}) and names: {present_target_names}")

        try:
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, labels=present_labels_encoded, average="weighted", zero_division=0)
            prec = precision_score(y_true, y_pred, labels=present_labels_encoded, average="weighted", zero_division=0)
            rec = recall_score(y_true, y_pred, labels=present_labels_encoded, average="weighted", zero_division=0)
            report = classification_report(y_true, y_pred, labels=present_labels_encoded, target_names=present_target_names, zero_division=0)
        except Exception as e:
            logger.error(f"Error calculating metrics for {name}: {e}")
            return {f"{name}_Error": f"Metrics calculation failed: {e}"}
            
        metrics_dict = {
            f"{name}_Accuracy": f"{acc*100:.2f}%", f"{name}_F1_Score": f"{f1:.4f}",
            f"{name}_Precision": f"{prec:.4f}", f"{name}_Recall": f"{rec:.4f}",
            f"{name}_Classification_Report": "\n" + report
        }
        print(f"\n📊 {name} METRICS:")
        print(f"| Accuracy: {acc*100:.2f}%\n| F1 Score: {f1:.4f}\n| Precision: {prec:.4f}\n| Recall: {rec:.4f}")
        print("\n📝 Classification Report:")
        print(report)
        logger.info(f"{name} METRICS: Acc={acc*100:.2f}%, F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")
        logger.info(f"Classification Report:\n{report}")
        return metrics_dict
    # --- End Dynamic Label Handling ---
        
    y_pred_best = best_clf.predict(Z_test)
    y_proba_best = None
    if hasattr(best_clf, "predict_proba"):
        try: y_proba_best = best_clf.predict_proba(Z_test)
        except Exception as e: logger.warning(f"Could not get probabilities from best classifier: {e}")
    best_metrics = calculate_and_store_metrics(y_test, y_pred_best, f"BestClassifier_{best_clf_name}", y_proba_best)
    all_metrics.update(best_metrics)
    
    y_pred_ensemble = None
    y_proba_ensemble = None
    try:
        y_pred_ensemble = ensemble.predict(Z_test)
        if hasattr(ensemble, "predict_proba"):
            try: y_proba_ensemble = ensemble.predict_proba(Z_test)
            except Exception as e: logger.warning(f"Could not get probabilities from ensemble: {e}")
    except Exception as e:
        logger.error(f"Error getting predictions from ensemble: {e}")
        y_pred_ensemble = None # Ensure it's None if prediction fails
        
    ensemble_metrics = calculate_and_store_metrics(y_test, y_pred_ensemble, "Ensemble", y_proba_ensemble)
    all_metrics.update(ensemble_metrics)
    
    save_metrics_to_file(all_metrics, results_filename)
    
    # --- Dynamic Label Handling for Plots (with checks) ---
    # Only plot if predictions were valid for the ensemble
    if y_pred_ensemble is not None and isinstance(y_pred_ensemble, np.ndarray) and len(y_pred_ensemble) == len(y_test):
        present_labels_encoded_test = np.unique(np.concatenate([y_test, y_pred_ensemble]))
        if subset_label_encoder is not None:
            try:
                present_labels_original_test = subset_label_encoder.inverse_transform(present_labels_encoded_test)
                present_target_names_test = [str(c) for c in present_labels_original_test]
            except Exception as e:
                logger.error(f"Error inverse transforming labels for plots: {e}. Using encoded labels.")
                present_labels_original_test = present_labels_encoded_test
                present_target_names_test = [str(c) for c in present_labels_encoded_test]
        else:
            present_labels_original_test = present_labels_encoded_test
            present_target_names_test = [str(c) for c in present_labels_original_test]
        num_present_classes_test = len(present_labels_encoded_test)
        
        cm_filename = f"results/{args.dataset.lower()}_confusion_matrix_ensemble.png"
        try:
            cm = confusion_matrix(y_test, y_pred_ensemble, labels=present_labels_encoded_test)
            plt.figure(figsize=(max(6, num_present_classes_test*0.8), max(5, num_present_classes_test*0.7))) # Dynamic size
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=present_target_names_test, yticklabels=present_target_names_test)
            plt.title(f"Confusion Matrix (Ensemble) - {args.dataset}", fontsize=12)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.savefig(cm_filename, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved confusion matrix to {cm_filename}")
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
            plt.close() # Ensure plot is closed even on error
    else:
        logger.warning("Skipping Confusion Matrix plot due to invalid ensemble predictions.")

    # Only plot ROC if probabilities were valid for the best classifier
    if y_proba_best is not None and isinstance(y_proba_best, np.ndarray) and y_proba_best.shape[0] == len(y_test):
        # Determine classes for binarization (all potential classes)
        if subset_label_encoder is not None:
            binarize_classes = subset_label_encoder.classes_
        else:
            binarize_classes = np.unique(np.concatenate([y_train, y_val, y_test]))
        num_binarize_classes = len(binarize_classes)
        
        if num_binarize_classes > 1:
            roc_filename = f"results/{args.dataset.lower()}_roc_curves_best_clf.png"
            try:
                y_test_bin = label_binarize(y_test, classes=binarize_classes)
                
                # Ensure probabilities match the binarized shape
                if y_test_bin.shape[1] != y_proba_best.shape[1]:
                     logger.warning(f"Shape mismatch between binarized labels ({y_test_bin.shape[1]}) and probabilities ({y_proba_best.shape[1]}). Adjusting proba columns for ROC.")
                     aligned_proba = np.zeros((y_proba_best.shape[0], num_binarize_classes))
                     clf_classes = best_clf.classes_ # Classes the classifier actually knows
                     for i, cls_orig in enumerate(binarize_classes):
                         try:
                             clf_idx = np.where(clf_classes == cls_orig)[0][0]
                             aligned_proba[:, i] = y_proba_best[:, clf_idx]
                         except IndexError:
                             # logger.warning(f"Class {cls_orig} not found in classifier output classes for ROC. Setting proba to 0.")
                             aligned_proba[:, i] = 0 # Assign 0 probability if class wasn't predicted
                     y_proba_best_aligned = aligned_proba
                else:
                     y_proba_best_aligned = y_proba_best
                     
                n_classes_plot = y_test_bin.shape[1]
                fpr, tpr, roc_auc = {}, {}, {}
                plt.figure(figsize=(8, 6))
                for i in range(n_classes_plot):
                    # Check if class i is actually present in y_test to avoid errors with label_binarize output
                    if i < y_test_bin.shape[1]:
                        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba_best_aligned[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])
                        label_name = str(binarize_classes[i])
                        plt.plot(fpr[i], tpr[i], lw=1.5, label=f"Cls {label_name} (AUC={roc_auc[i]:.2f})")
                    else:
                        logger.warning(f"Skipping ROC curve for class index {i} as it exceeds binarized label dimensions.")
                        
                plt.plot([0, 1], [0, 1], "k--", lw=1.5)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC (Best: {best_clf_name}) - {args.dataset}", fontsize=12)
                plt.legend(loc="lower right", fontsize="small")
                plt.savefig(roc_filename, dpi=150, bbox_inches="tight")
                plt.close()
                logger.info(f"Saved ROC curves to {roc_filename}")
            except Exception as e:
                logger.error(f"Error plotting ROC curves: {e}")
                plt.close()
        else:
             logger.warning("Skipping ROC plot: Only one class present after binarization.")
    else:
        logger.warning("Skipping ROC plot due to invalid best classifier probabilities.")
        
    # t-SNE plot (no dependency on predictions)
    tsne_filename = f"results/{args.dataset.lower()}_tsne_embeddings.png"
    tsne = TSNE(n_components=2, perplexity=min(30, Z_test.shape[0]-1), n_iter=500, random_state=42)
    if Z_test.shape[0] > 1:
        try:
            Z_2d = tsne.fit_transform(Z_test)
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(Z_2d[:, 0], Z_2d[:, 1], c=y_test, cmap="viridis", s=15, alpha=0.7)
            # Use original names for colorbar ticks
            present_labels_encoded_tsne = np.unique(y_test)
            if subset_label_encoder is not None:
                try:
                    present_labels_original_tsne = subset_label_encoder.inverse_transform(present_labels_encoded_tsne)
                    present_target_names_tsne = [str(c) for c in present_labels_original_tsne]
                except Exception as e:
                    logger.error(f"Error inverse transforming labels for t-SNE colorbar: {e}. Using encoded labels.")
                    present_target_names_tsne = [str(c) for c in present_labels_encoded_tsne]
            else:
                present_target_names_tsne = [str(c) for c in present_labels_encoded_tsne]
            colorbar_ticks = present_labels_encoded_tsne
            colorbar_labels = present_target_names_tsne
            # Create a mapping from tick value to label name for the formatter
            tick_label_map = {tick: label for tick, label in zip(colorbar_ticks, colorbar_labels)}
            plt.colorbar(scatter, ticks=colorbar_ticks, format=plt.FuncFormatter(lambda val, loc: tick_label_map.get(int(val), "")))
            plt.title(f"t-SNE Embeddings (Test Set) - {args.dataset}", fontsize=12)
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.savefig(tsne_filename, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved t-SNE plot to {tsne_filename}")
        except Exception as e:
            logger.error(f"Error plotting t-SNE: {e}")
            plt.close()
    else:
        logger.warning("Skipping t-SNE plot due to insufficient test data points.")
    # --- End Dynamic Label Handling for Plots ---
        
    print_stage_footer("LINEAR PROBING", args.dataset)
    return ensemble

def evaluate_alignment_with_mh(X, y, user_ids, encoder, args):
    global subset_label_encoder # Access the global encoder
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
    n_splits = min(3, len(unique_users) // 2)
    if n_splits == 0:
        logger.warning("Skipping alignment evaluation: Not enough users for splits.")
        print_stage_footer("ALIGNMENT EVALUATION", args.dataset)
        return 0, 0, 0
    accuracies, f1_scores, temporal_coherences = [], [], []
    for split in range(n_splits):
        np.random.shuffle(unique_users)
        split_idx = len(unique_users) // 2
        source_users, target_users = unique_users[:split_idx], unique_users[split_idx:]
        source_mask, target_mask = np.isin(user_ids, source_users), np.isin(user_ids, target_users)
        Z_source, Z_target = Z[source_mask], Z[target_mask]
        y_source, y_target = y[source_mask], y[target_mask]
        if Z_source.shape[0] == 0 or Z_target.shape[0] == 0:
            logger.warning(f"Skipping split {split+1}: Empty source or target set.")
            continue
        aligner = QuantumHungarian(temp=ALIGNMENT_TEMP, n_iter=ALIGNMENT_N_ITER, feature_weight=args.feature_weight)
        try:
            row_ind, col_ind = aligner(Z_source, Z_target)
            # Pass original labels if available for evaluation clarity
            y_source_orig = subset_label_encoder.inverse_transform(y_source) if subset_label_encoder else y_source
            y_target_orig = subset_label_encoder.inverse_transform(y_target) if subset_label_encoder else y_target
            acc, f1, temporal = evaluate_alignment(y_source_orig, y_target_orig, row_ind, col_ind)
            logger.info(f"Split {split+1}/{n_splits}, QuantumHungarian - Acc: {acc*100:.2f}%, F1: {f1:.4f}, Temporal: {temporal:.4f}")
            accuracies.append(acc)
            f1_scores.append(f1)
            temporal_coherences.append(temporal)
        except Exception as e:
             logger.error(f"Error during alignment in split {split+1}: {e}")
             accuracies.append(0)
             f1_scores.append(0)
             temporal_coherences.append(0)
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
        "Average_Accuracy": f"{avg_acc*100:.2f}%", "Average_F1_Score": f"{avg_f1:.4f}",
        "Average_Temporal_Coherence": f"{avg_temporal:.4f}", "Number_of_Valid_Splits": len(accuracies)
    }
    results_filename = f"results/{args.dataset.lower()}_alignment_metrics.txt"
    save_metrics_to_file(alignment_metrics, results_filename)
    print_stage_footer("ALIGNMENT EVALUATION", args.dataset)
    return avg_acc, avg_f1, avg_temporal

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    elif torch.backends.mps.is_available():
        if torch.backends.mps.is_built(): return torch.device("mps")
        else: logger.warning("MPS available but not built. Falling back to CPU."); return torch.device("cpu")
    else: return torch.device("cpu")

# ---------------------- Main Function ----------------------
def main():
    global subset_label_encoder # Allow modification of the global variable
    parser = argparse.ArgumentParser(description="Enhanced HAR SSL Pipeline (v6 - Lightweight v4)")
    parser.add_argument("--mode", type=str, default="all", choices=["pretrain", "finetune", "linear_probe", "alignment", "all"], help="Pipeline mode")
    parser.add_argument("--dataset", type=str, default="PAMAP2", choices=["UCI", "PAMAP2"], help="Dataset to use") # Add future datasets here
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent dimension size")
    parser.add_argument("--transformer_seq_len", type=int, default=32, help="Sequence length for transformer input")
    parser.add_argument("--num_transformer_layers", type=int, default=1, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--window_size", type=int, default=192, help="Window size")
    parser.add_argument("--window_step", type=int, default=64, help="Step size")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs for SSL pretraining")
    parser.add_argument("--ft_epochs", type=int, default=1, help="Epochs for fine-tuning")
    parser.add_argument("--lr", type=float, default=0.0002, help="LR for SSL pretraining")
    parser.add_argument("--ft_lr", type=float, default=0.0005, help="LR for fine-tuning")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--num_workers", type=int, default=0, help="Data loading workers")
    parser.add_argument("--ssl_loss", type=str, default="contrastive", choices=["contrastive", "vicreg"], help="SSL loss function")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temp for contrastive loss")
    parser.add_argument("--sim_weight", type=float, default=25.0, help="Sim weight for VICReg")
    parser.add_argument("--var_weight", type=float, default=25.0, help="Var weight for VICReg")
    parser.add_argument("--cov_weight", type=float, default=1.0, help="Cov weight for VICReg")
    parser.add_argument("--time_warp_limit", type=float, default=0.2, help="Time warping limit")
    parser.add_argument("--channel_drop_prob", type=float, default=0.2, help="Channel dropout probability")
    parser.add_argument("--noise_std", type=float, default=0.1, help="Noise standard deviation")
    parser.add_argument("--scale_min", type=float, default=0.8, help="Minimum scaling factor")
    parser.add_argument("--scale_max", type=float, default=1.2, help="Maximum scaling factor")
    parser.add_argument("--feature_weight", type=float, default=0.7, help="Feature weight for QuantumHungarian")
    parser.add_argument("--subset_fraction", type=float, default=1.0, help="Fraction of data to use (e.g., 0.1 for 10%)")
    
    args = parser.parse_args()
    logger.info(f"Starting pipeline (v6 - Lightweight v4) with arguments: {vars(args)}")
    os.makedirs("results", exist_ok=True)
    logger.info("Created/verified \'results\' directory for outputs.")
    
    # --- Generalized Data Loading ---
    # The load_data function (in utils.py) should handle dataset-specific logic based on args.dataset
    # Here, we just call it and expect the correct format back.
    logger.info(f"Loading {args.dataset} dataset...")
    try:
        # Expect load_data to return raw data first if windowing is needed later (like UCI)
        # or windowed data directly if pre-windowed (like PAMAP2 in this example)
        loaded_data = load_data(args.dataset)
        
        # Check format and apply windowing if necessary
        if args.dataset == "UCI": # Example: UCI needs windowing after loading
            X_train_raw, y_train_raw, X_test_raw, y_test_raw, user_train_raw, user_test_raw = loaded_data
            # Split train into train/val
            X_train_raw, X_val_raw, y_train_raw, y_val_raw, user_train_raw, user_val_raw = train_test_split(
                X_train_raw, y_train_raw, user_train_raw, test_size=0.2, stratify=y_train_raw, random_state=42
            )
            logger.info("Applying sliding window to UCI data...")
            X_train, y_train, user_train = sliding_window(X_train_raw, args.window_size, args.window_step, y_train_raw, user_train_raw)
            X_val, y_val, user_val = sliding_window(X_val_raw, args.window_size, args.window_step, y_val_raw, user_val_raw)
            X_test, y_test, user_test = sliding_window(X_test_raw, args.window_size, args.window_step, y_test_raw, user_test_raw)
            logger.info(f"UCI data after windowing: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
        elif args.dataset == "PAMAP2": # Example: PAMAP2 load_data returns windowed data
            X_train, y_train, X_val, y_val, X_test, y_test, user_train, user_val, user_test = loaded_data
            logger.info(f"PAMAP2 data loaded: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
        else:
            # Add logic here for future datasets
            # Assume loaded_data is in the final windowed format for now
            logger.warning(f"Dataset '{args.dataset}' not explicitly handled for windowing. Assuming pre-windowed data.")
            X_train, y_train, X_val, y_val, X_test, y_test, user_train, user_val, user_test = loaded_data
            
    except Exception as e:
        logger.error(f"Failed to load or process dataset '{args.dataset}': {e}")
        raise
    # --- End Generalized Data Loading ---
        
    if len(X_train.shape) != 3:
        raise ValueError(f"X_train must be 3D after loading/windowing. Got shape: {X_train.shape}")
        
    # Apply subset fraction if specified
    if args.subset_fraction < 1.0:
        logger.warning(f"Using only {args.subset_fraction*100:.1f}% of the data for lightweight testing.")
        def take_subset(X, y, users, fraction):
            num_samples = int(len(X) * fraction)
            if num_samples == 0: return X[:1], y[:1], users[:1] # Ensure at least one sample
            indices = np.random.choice(len(X), num_samples, replace=False)
            return X[indices], y[indices], users[indices]
        
        X_train, y_train, user_train = take_subset(X_train, y_train, user_train, args.subset_fraction)
        X_val, y_val, user_val = take_subset(X_val, y_val, user_val, args.subset_fraction)
        X_test, y_test, user_test = take_subset(X_test, y_test, user_test, args.subset_fraction)
        logger.info(f"Subset sizes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

        # --- Add Label Re-encoding ---
        all_subset_labels = np.concatenate([y_train, y_val, y_test])
        unique_subset_labels = np.unique(all_subset_labels)
        
        if len(unique_subset_labels) == 0:
            raise ValueError("No labels found in the data subset!")
            
        subset_label_encoder = LabelEncoder()
        subset_label_encoder.fit(unique_subset_labels) # Fit on original labels present in subset
        
        # Transform labels to 0-based contiguous integers
        y_train = subset_label_encoder.transform(y_train)
        y_val = subset_label_encoder.transform(y_val)
        y_test = subset_label_encoder.transform(y_test)
        
        num_classes_subset = len(subset_label_encoder.classes_)
        logger.info(f"Re-encoded labels for subset. Original unique labels in subset: {unique_subset_labels}. New number of classes: {num_classes_subset}")
        # --- End Label Re-encoding ---
    else:
        subset_label_encoder = None # Ensure it's None if not using subset

    encoder = None
    if args.mode == "pretrain" or args.mode == "all":
        encoder = train_ssl(X_train, X_val, y_val, args)
    if encoder is None:
        encoder_path = f"results/ssl_encoder_{args.dataset.lower()}.pth"
        if not os.path.exists(encoder_path):
             encoder_path = f"results/best_ssl_encoder_{args.dataset.lower()}.pth"
        if os.path.exists(encoder_path):
            logger.info(f"Loading pre-trained encoder from {encoder_path}")
            # Determine input dim dynamically from loaded data
            input_dim = X_train.shape[2]
            encoder = SimplifiedHybridEncoder(
                input_dim, args.latent_dim, dropout=args.dropout,
                transformer_seq_len=args.transformer_seq_len,
                num_transformer_layers=args.num_transformer_layers
            ).to(get_device())
            try:
                encoder.load_state_dict(torch.load(encoder_path, map_location=get_device()))
            except Exception as e:
                 logger.error(f"Failed to load encoder state dict from {encoder_path}: {e}")
                 raise
        elif args.mode != "pretrain":
             logger.error(f"Pre-trained encoder not found at {encoder_path} and mode is not 'pretrain'. Please run pretraining first.")
             raise FileNotFoundError(f"Encoder not found: {encoder_path}")
    if encoder is None and args.mode != "pretrain":
         logger.error("Encoder is required for downstream tasks but was not loaded or trained.")
         return
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
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")


