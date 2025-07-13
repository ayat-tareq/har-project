# train.py
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from utils import load_data, sliding_window, QuantumHungarian, evaluate_alignment
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
    def __init__(self, input_dim, latent_dim=256):
        super().__init__()
        self.conv_branch = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64)
        )
        self.pos_enc = PositionalEncoding(d_model=128)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512, dropout=0.1, batch_first=True),
            num_layers=2
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*64, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_branch(x)
        x = x.permute(0, 2, 1)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.projector(x)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
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
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    contrastive = ContrastiveLoss()

    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            x = batch[0].to(device)
            x1 = x + 0.1 * torch.randn_like(x)
            x2 = x + 0.1 * torch.randn_like(x)
            with autocast(enabled=(device.type == 'cuda')):
                z1 = model(x1)
                z2 = model(x2)
                loss = contrastive(z1, z2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_loss += loss.item()

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

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Z, y)
    acc = clf.score(Z, y)
    print(f"[Probe] Accuracy: {acc*100:.2f}%")

    tsne = TSNE(n_components=2, perplexity=30)
    Z_2d = tsne.fit_transform(Z)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(Z_2d[:, 0], Z_2d[:, 1], c=y, cmap='tab20', s=20, alpha=0.8)
    plt.colorbar(scatter)
    plt.title("t-SNE Embeddings After SSL + Probe")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.savefig("tsne_t-sne_embeddings_after_ssl_+_probe.png")
    print("Saved t-SNE plot to tsne_t-sne_embeddings_after_ssl_+_probe.png")

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
    row, col = QuantumHungarian()(Z, Z[shuffled])
    acc = evaluate_alignment(y, y[shuffled], row, col)
    print(f"[Alignment] Accuracy: {acc*100:.2f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['pretrain', 'linear_probe', 'align'], required=True)
    parser.add_argument('--dataset', choices=['UCI', 'PAMAP2'], required=True)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--window_size', type=int, default=128)
    parser.add_argument('--window_step', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
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
