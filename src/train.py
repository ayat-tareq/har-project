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
from utils import load_data, sliding_window, SinkhornAlign, evaluate_alignment
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input, got shape {x.shape}")
        x = x.permute(0, 2, 1)
        x = self.conv_branch(x)
        x = x.permute(0, 2, 1)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.projector(x)

class LinearClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim_matrix = torch.mm(z1, z2.T) / self.temperature
        targets = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(sim_matrix, targets)

def augment_batch(x):
    noise = 0.02 * torch.randn_like(x)
    scale = torch.randn(x.size(0), 1, x.size(2), device=x.device) * 0.1 + 1.0
    return (x + noise) * scale

def visualize_tsne(embeddings, labels, title="t-SNE Visualization"):
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10', alpha=0.6, edgecolors='k')
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"tsne_{title.replace(' ', '_').lower()}.png")
    print(f"Saved t-SNE plot to tsne_{title.replace(' ', '_').lower()}.png")
    plt.close()

def ssl_pretrain(X, args):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if len(X.shape) == 2:
        X, _ = sliding_window(X, args.window_size, args.window_step)

    loader = DataLoader(torch.tensor(X, dtype=torch.float32), batch_size=args.batch_size, shuffle=True)
    model = HybridEncoder(input_dim=X.shape[2], latent_dim=args.latent_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    contrast = ContrastiveLoss()

    model.train()
    for epoch in range(args.epochs):
        total = 0
        for xb in loader:
            xb = xb.to(device)
            x1 = augment_batch(xb)
            x2 = augment_batch(xb)
            with autocast(enabled=(device.type == 'cuda')):
                z1 = model(x1)
                z2 = model(x2)
                loss = contrast(z1, z2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total += loss.item()

        print(f"[SSL] Epoch {epoch+1}/{args.epochs} - Loss: {total/len(loader):.4f}")

    torch.save(model.state_dict(), 'ssl_encoder.pth')
    print("[SSL] Saved encoder to ssl_encoder.pth")

def linear_probe(X, y, args):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if len(X.shape) == 2:
        X, y = sliding_window(X, args.window_size, args.window_step, y)

    encoder = HybridEncoder(input_dim=X.shape[2], latent_dim=args.latent_dim).to(device)
    encoder.load_state_dict(torch.load('ssl_encoder.pth'))
    encoder.eval()

    classifier = LinearClassifier(args.latent_dim, len(set(y))).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        X_embed = encoder(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()

    dataset = TensorDataset(torch.tensor(X_embed, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    classifier.train()
    for epoch in range(50):
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = classifier(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()

        print(f"[Probe] Epoch {epoch+1}/50 - Loss: {total/len(loader):.4f}")

    classifier.eval()
    preds = classifier(torch.tensor(X_embed, dtype=torch.float32).to(device))
    preds = torch.argmax(preds, dim=1).cpu().numpy()
    acc = accuracy_score(y, preds)
    print(f"[Probe] Accuracy: {acc*100:.2f}%")
    visualize_tsne(X_embed, y, title="t-SNE Embeddings After SSL + Probe")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['pretrain', 'linear_probe'], required=True)
    parser.add_argument('--dataset', choices=['UCI', 'PAMAP2'], required=True)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--window_size', type=int, default=128)
    parser.add_argument('--window_step', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    X_train, y_train, _, _ = load_data(args.dataset)

    if args.mode == 'pretrain':
        ssl_pretrain(X_train, args)
    elif args.mode == 'linear_probe':
        linear_probe(X_train, y_train, args)

if __name__ == '__main__':
    main()
