# train.py (Enhanced Version)
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from utils import load_data, sliding_window, SinkhornAlign, evaluate_alignment
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import joblib

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class HybridEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=512):  # Increased latent dimension
        super().__init__()
        self.conv_branch = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=5, padding=2),  # More filters
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(64)
        )
        self.pos_enc = PositionalEncoding(d_model=256)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.2,
                batch_first=True
            ),  # Larger transformer
            num_layers=3
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*64, 1024),  # Expanded projection
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, latent_dim)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_branch(x)
        x = x.permute(0, 2, 1)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.projector(x)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):  # Lower temperature
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        logits = torch.mm(z1, z2.T) / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(logits, labels)

def train_ssl(X, args):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    if len(X.shape) == 2:
        X, _ = sliding_window(X, args.window_size, args.window_step)

    model = HybridEncoder(X.shape[2], args.latent_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)  # Added weight decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)  # Learning rate scheduling
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    contrastive = ContrastiveLoss()

    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=args.batch_size, shuffle=True, pin_memory=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            x = batch[0].to(device)
            # Enhanced augmentations
            x1 = x + torch.normal(0, 0.2, size=x.shape, device=device)  # Stronger noise
            x2 = x * torch.FloatTensor(x.shape).uniform_(0.8, 1.2).to(device)  # Scaling augmentation
            
            with autocast(enabled=(device.type == 'cuda')):
                z1 = model(x1)
                z2 = model(x2)
                loss = contrastive(z1, z2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_loss += loss.item()
        
        scheduler.step()
        print(f"[SSL] Epoch {epoch+1}/{args.epochs} - Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "ssl_encoder.pth")
    print("[SSL] Saved encoder to ssl_encoder.pth")

def linear_probe(X, y, args):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if len(X.shape) == 2:
        X, y = sliding_window(X, args.window_size, args.window_step, y)

    model = HybridEncoder(X.shape[2], args.latent_dim).to(device)
    model.load_state_dict(torch.load("ssl_encoder.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        Z = model(X_tensor).cpu().numpy()

    # Enhanced classifier with better params
    clf = LogisticRegression(
        max_iter=5000,
        C=0.1,
        solver='saga',
        penalty='l2',
        class_weight='balanced',
        random_state=42
    )
    clf.fit(Z, y)
    acc = clf.score(Z, y)
    print(f"[Probe] Accuracy: {acc*100:.2f}%")

    # Enhanced visualization
    tsne = TSNE(n_components=2, perplexity=40, n_iter=3000)
    Z_2d = tsne.fit_transform(Z)
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(Z_2d[:, 0], Z_2d[:, 1], c=y, cmap='viridis', s=25, alpha=0.9)
    plt.colorbar(scatter, ticks=np.unique(y))
    plt.title("Enhanced t-SNE Visualization of SSL Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig("tsne_embeddings.png", dpi=300)
    print("Saved enhanced t-SNE plot to tsne_embeddings.png")

def evaluate_alignment_with_mh(X, y, args):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if len(X.shape) == 2:
        X, y = sliding_window(X, args.window_size, args.window_step, y)

    model = HybridEncoder(X.shape[2], args.latent_dim).to(device)
    model.load_state_dict(torch.load("ssl_encoder.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        Z = model(X_tensor).cpu().numpy()

    shuffled = np.random.permutation(len(Z))
    row, col = SinkhornAlign()(Z, Z[shuffled])
    acc = evaluate_alignment(y, y[shuffled], row, col)
    print(f"[Alignment] Accuracy: {acc*100:.2f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['pretrain', 'linear_probe', 'align'], required=True)
    parser.add_argument('--dataset', choices=['UCI', 'PAMAP2'], required=True)
    parser.add_argument('--latent_dim', type=int, default=512)  # Updated default
    parser.add_argument('--window_size', type=int, default=128)
    parser.add_argument('--window_step', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)  # Increased default epochs
    parser.add_argument('--lr', type=float, default=2e-4)  # Adjusted learning rate
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_data(args.dataset)

    if args.mode == 'pretrain':
        train_ssl(X_train, args)
    elif args.mode == 'linear_probe':
        linear_probe(X_train, y_train, args)
    elif args.mode == 'align':
        evaluate_alignment_with_mh(X_test, y_test, args)

if __name__ == "__main__":
    main()
