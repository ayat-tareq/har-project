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
from utils import load_data, SinkhornAlign, evaluate_alignment
from sklearn.metrics import confusion_matrix, f1_score

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

class HybridLoss(nn.Module):
    def __init__(self, contrast_temp=0.5):
        super().__init__()
        self.temp = contrast_temp

    def forward(self, proj1, proj2):
        proj1 = F.normalize(proj1, dim=1)
        proj2 = F.normalize(proj2, dim=1)
        logits = torch.mm(proj1, proj2.T) / self.temp
        targets = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, targets)

def train_ssl(X_train, y_train, args):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.set_float32_matmul_precision('medium')

    X_windows, y_windows = X_train, y_train
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_windows, dtype=torch.float32),
                      torch.tensor(y_windows, dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=True
    )

    model = HybridEncoder(input_dim=X_windows.shape[2], latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    criterion = HybridLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            x1 = x + 0.1 * torch.randn_like(x)
            x2 = x + 0.1 * torch.randn_like(x)

            with autocast(enabled=(device.type == 'cuda')):
                proj1 = model(x1)
                proj2 = model(x2)
                loss = criterion(proj1, proj2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {total_loss/len(train_loader):.4f}")

    return model

def evaluate(model, X_test, y_test, args):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        emb = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
        shuffled_idx = np.random.permutation(len(emb))
        row, col = SinkhornAlign()(emb, emb[shuffled_idx])
        acc = evaluate_alignment(y_test, y_test[shuffled_idx], row, col)
        print(f"Final Alignment Accuracy: {acc*100:.2f}%")
        cm = confusion_matrix(y_test[row], y_test[shuffled_idx][col])
        print("Confusion Matrix:\n", cm)
        print("Macro F1:", f1_score(y_test[row], y_test[shuffled_idx][col], average='macro'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['UCI', 'PAMAP2'], required=True)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--window_size', type=int, default=128)
    parser.add_argument('--window_step', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_data(args.dataset)
    model = train_ssl(X_train, y_train, args)
    evaluate(model, X_test, y_test, args)

if __name__ == "__main__":
    main()
