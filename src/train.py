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
from sklearn.metrics import (
    confusion_matrix, f1_score, accuracy_score,
    precision_score, recall_score, roc_curve, auc,
    classification_report
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------------------- Utility Functions ----------------------
def print_stage_header(stage_name, dataset):
    print(f"\n{'-'*60}")
    print(f"🚀 STARTING {stage_name.upper()} - DATASET: {dataset}")
    print(f"{'-'*60}")

def print_stage_footer(stage_name, dataset):
    print(f"\n{'-'*60}")
    print(f"✅ COMPLETED {stage_name.upper()} - DATASET: {dataset}")
    print(f"{'-'*60}\n")

# ---------------------- Model Components ----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))  # Fixed here
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class HybridEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=512):
        super().__init__()
        self.conv_branch = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=5, padding=2),
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
            ),
            num_layers=3
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*64, 1024),
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
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        logits = torch.mm(z1, z2.T) / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(logits, labels)

# ---------------------- Training & Evaluation ----------------------
def train_ssl(X, args):
    print_stage_header("SSL PRETRAINING", args.dataset)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    if len(X.shape) == 2:
        X, _ = sliding_window(X, args.window_size, args.window_step)

    model = HybridEncoder(X.shape[2], args.latent_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    contrastive = ContrastiveLoss()

    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=args.batch_size, shuffle=True, pin_memory=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            x = batch[0].to(device)
            x1 = x + torch.normal(0, 0.2, size=x.shape, device=device)
            x2 = x * torch.FloatTensor(x.shape).uniform_(0.8, 1.2).to(device)
            
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
        print(f"[Epoch {epoch+1}/{args.epochs}] Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "ssl_encoder.pth")
    print("\n💾 Saved SSL encoder to ssl_encoder.pth")
    print_stage_footer("SSL PRETRAINING", args.dataset)

def linear_probe(X_train, y_train, X_test, y_test, args):
    print_stage_header("LINEAR PROBING", args.dataset)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    def process_data(X, y=None):
        if len(X.shape) == 2:
            return sliding_window(X, args.window_size, args.window_step, y)
        return X, y

    X_train, y_train = process_data(X_train, y_train)
    X_test, y_test = process_data(X_test, y_test)

    model = HybridEncoder(X_train.shape[2], args.latent_dim).to(device)
    try:
        model.load_state_dict(torch.load("ssl_encoder.pth", map_location=device))
    except RuntimeError as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")
    model.eval()

    with torch.no_grad():
        Z_train = model(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu().numpy()
        Z_test = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

    clf = LogisticRegression(
        max_iter=5000,
        C=0.1,
        solver='saga',
        penalty='l2',
        class_weight='balanced',
        random_state=42
    )
    clf.fit(Z_train, y_train)
    joblib.dump(clf, f"classifier_{args.dataset.lower()}.pkl")

    # Train predictions
    y_pred_train = clf.predict(Z_train)
    y_proba_train = clf.predict_proba(Z_train)

    # Test predictions
    y_pred_test = clf.predict(Z_test)
    y_proba_test = clf.predict_proba(Z_test)

    # Calculate metrics
    def print_metrics(y_true, y_pred, y_proba, set_name):
        print(f"\n📊 {set_name.upper()} METRICS:")
        print(f"| Accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%")
        print(f"| F1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")
        print(f"| Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
        print(f"| Recall: {recall_score(y_true, y_pred, average='weighted'):.4f}")
        print("\n📝 Classification Report:")
        print(classification_report(y_true, y_pred))

    print_metrics(y_train, y_pred_train, y_proba_train, "train")
    print_metrics(y_test, y_pred_test, y_proba_test, "test")

    # Confusion Matrix
    classes = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {args.dataset}', fontsize=14)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f"confusion_matrix_{args.dataset.lower()}.png", dpi=300)
    print(f"\n💡 Saved confusion matrix to confusion_matrix_{args.dataset.lower()}.png")

    # ROC Curve
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = y_test_bin.shape[1]
    
    fpr, tpr, roc_auc = {}, {}, {}
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba_test[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f'Class {classes[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {args.dataset}', fontsize=14)
    plt.legend(loc="lower right")
    plt.savefig(f"roc_curves_{args.dataset.lower()}.png", dpi=300)
    print(f"💡 Saved ROC curves to roc_curves_{args.dataset.lower()}.png")

    # t-SNE Visualization
    tsne = TSNE(n_components=2, perplexity=40, n_iter=3000, random_state=42)
    Z_combined = np.vstack([Z_train, Z_test])
    Z_2d = tsne.fit_transform(Z_combined)
    
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(Z_2d[:, 0], Z_2d[:, 1],
                         c=np.concatenate([y_train, y_test]),
                         cmap='viridis', s=25, alpha=0.6)
    plt.colorbar(scatter, ticks=np.unique(y_test))
    plt.title(f"t-SNE Embeddings - {args.dataset}", fontsize=16)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig(f"tsne_embeddings_{args.dataset.lower()}.png", dpi=300)
    print(f"💡 Saved t-SNE plot to tsne_embeddings_{args.dataset.lower()}.png")

    print_stage_footer("LINEAR PROBING", args.dataset)

def evaluate_alignment_with_mh(X, y, args):
    print_stage_header("ALIGNMENT EVALUATION", args.dataset)
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
    print(f"\n🎯 Alignment Accuracy: {acc*100:.2f}%")
    print_stage_footer("ALIGNMENT EVALUATION", args.dataset)

# ---------------------- Main Function ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str,
                       choices=['pretrain', 'linear_probe', 'align'],
                       required=True)
    parser.add_argument('--dataset', choices=['UCI', 'PAMAP2'], required=True)
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--window_size', type=int, default=128)
    parser.add_argument('--window_step', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_data(args.dataset)

    if args.mode == 'pretrain':
        train_ssl(X_train, args)
    elif args.mode == 'linear_probe':
        linear_probe(X_train, y_train, X_test, y_test, args)
    elif args.mode == 'align':
        evaluate_alignment_with_mh(X_test, y_test, args)

if __name__ == "__main__":
    main()
