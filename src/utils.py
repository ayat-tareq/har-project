# utils.py
import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

def load_data(dataset='UCI'):
    if dataset == 'UCI':
        return load_uci()
    elif dataset == 'PAMAP2':
        return load_pamap2()
    raise ValueError(f"Unsupported dataset: {dataset}")

def load_uci(data_root='data/UCI_HAR'):
    def read_signals(folder):
        signals = []
        sensor_axes = ['body_acc', 'body_gyro']
        axes = ['x', 'y', 'z']
        for sensor in sensor_axes:
            for axis in axes:
                path = os.path.join(folder, 'Inertial Signals', f'{sensor}_{axis}_{"train" if "train" in folder else "test"}.txt')
                signals.append(np.loadtxt(path))
        return np.concatenate(signals, axis=1)

    X_train = read_signals(f'{data_root}/train')
    X_test = read_signals(f'{data_root}/test')
    y_train = np.loadtxt(f'{data_root}/train/y_train.txt').astype(int)
    y_test = np.loadtxt(f'{data_root}/test/y_test.txt').astype(int)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test

def load_pamap2(data_root='data/PAMAP2/PAMAP2_Dataset/Protocol', window_size=192):
    segments = defaultdict(list)
    for file in os.listdir(data_root):
        if file.endswith('.dat'):
            df = pd.read_csv(os.path.join(data_root, file), delim_whitespace=True, header=None)
            df = df[df[1].between(1, 24)]
            df = df.dropna()
            current_label = None
            buffer = []
            for _, row in df.iterrows():
                label = int(row[1])
                if label != current_label:
                    if current_label is not None and len(buffer) >= window_size:
                        segments[current_label].append(np.stack(buffer))
                    buffer = []
                    current_label = label
                buffer.append(row[[2] + list(range(4, 40))].values.astype(np.float32))
            if current_label is not None and len(buffer) >= window_size:
                segments[current_label].append(np.stack(buffer))

    X, y = [], []
    for label, seqs in segments.items():
        for seq in seqs:
            for i in range(0, len(seq) - window_size + 1, window_size // 2):
                window = seq[i:i+window_size]
                if window.shape[0] == window_size:
                    X.append(window)
                    y.append(label)

    X = np.stack(X)
    y = np.array(y, dtype=np.int32)
    scaler = StandardScaler()
    N, W, D = X.shape
    X = X.reshape(-1, D)
    X = scaler.fit_transform(X)
    X = X.reshape(N, W, D)

    split = int(0.7 * len(X))
    return X[:split], y[:split], X[split:], y[split:]

def sliding_window(data, window_size, step, labels=None):
    n_samples = (len(data) - window_size) // step + 1
    windows = []
    label_windows = []

    for i in range(n_samples):
        data_window = data[i * step:i * step + window_size]
        if data_window.shape[0] == window_size:
            windows.append(data_window)
            if labels is not None:
                label_window = labels[i * step:i * step + window_size]
                if label_window.shape[0] == window_size and (label_window >= 1).all():
                    label_windows.append(label_window.astype(np.int32))

    windows = np.stack(windows)
    if labels is not None and label_windows:
        label_windows = np.stack(label_windows)
        majority = np.apply_along_axis(lambda x: np.bincount(x, minlength=100).argmax(), 1, label_windows)
        return windows, majority
    elif labels is not None:
        print("[Warning] No valid label windows were found.")
        return windows, np.zeros(len(windows), dtype=np.int32)
    return windows, None

class SinkhornAlign:
    def __init__(self, n_iter=20, epsilon=0.1):
        self.n_iter = n_iter
        self.epsilon = epsilon

    def __call__(self, A, B):
        A = np.nan_to_num(A)
        B = np.nan_to_num(B)

        dist = np.linalg.norm(A[:, None] - B[None, :], axis=2)
        K = np.exp(-dist ** 2 / self.epsilon) + 1e-9
        u = np.ones(K.shape[0]) / K.shape[0]
        v = np.ones(K.shape[1]) / K.shape[1]

        for _ in range(self.n_iter):
            u = 1.0 / (K @ v + 1e-8)
            v = 1.0 / (K.T @ u + 1e-8)

        P = np.diag(u) @ K @ np.diag(v)
        row_ind, col_ind = linear_sum_assignment(-P)
        return row_ind, col_ind

def evaluate_alignment(y_true, y_shuffled, row_ind, col_ind):
    if len(row_ind) == 0 or len(col_ind) == 0:
        return 0.0
    y_true = np.array(y_true)
    y_shuffled = np.array(y_shuffled)
    correct = (y_true[row_ind] == y_shuffled[col_ind]).astype(int)
    return correct.mean() if len(correct) > 0 else 0.0
