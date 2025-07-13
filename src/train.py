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
    def __init__(self, input_dim, latent_dim=256, num_classes=25):
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
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x, return_embedding=False):
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input, got shape {x.shape}")
        x = x.permute(0, 2, 1)
        x = self.conv_branch(x)
        x = x.permute(0, 2, 1)
        x = self.pos_enc(x)
        x = self.transformer(x)
        z = self.projector(x)
        if return_embedding:
            return z
        return self.classifier(z)

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

def jitter_scale_mask(x):
    jitter = x + 0.02 * torch.randn_like(x)
    scale = torch.randn(x.size(0), 1, x.size(2), device=x.device) * 0.1 + 1.0
    mask = torch.bernoulli(torch.full(x.shape, 0.95, device=x.device))
    return jitter * scale * mask

def train_ssl(X_train, y_train, args):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.set_float32_matmul_precision('medium')

    # Fix shape for UCI data
    if len(X_train.shape) == 2:
        X_train, y_train = sliding_window(X_train, args.window_size, args.window_step, y_train)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=True
    )

    model = HybridEncoder(input_dim=X_train.shape[2], latent_dim=args.latent_dim, num_classes=len(set(y_train))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    contrastive = HybridLoss()
    ce_loss = nn.CrossEntropyLoss()

    warmup_epochs = 10

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            x1 = jitter_scale_mask(x)
            x2 = jitter_scale_mask(x)

            with autocast(enabled=(device.type == 'cuda')):
                z1 = model(x1, return_embedding=True)
                z2 = model(x2, return_embedding=True)
                pred = model(x, return_embedding=False)

                if epoch < warmup_epochs:
                    loss = ce_loss(pred, y)
                else:
                    loss = 0.7 * contrastive(z1, z2) + 0.3 * ce_loss(pred, y)

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
        if len(X_test.shape) == 2:
            X_test, y_test = sliding_window(X_test, args.window_size, args.window_step, y_test)

        x_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        embeddings = model(x_tensor, return_embedding=True).cpu().numpy()
        pred_logits = model(x_tensor, return_embedding=False).cpu()
        preds = torch.argmax(pred_logits, dim=1).numpy()

        acc_cls = accuracy_score(y_test, preds)
        print(f"Classifier Accuracy: {acc_cls*100:.2f}%")

        shuffled_idx = np.random.permutation(len(embeddings))
        row, col = SinkhornAlign()(embeddings, embeddings[shuffled_idx])
        acc_align = evaluate_alignment(y_test, y_test[shuffled_idx], row, col)
        print(f"Final Alignment Accuracy: {acc_align*100:.2f}%")

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
