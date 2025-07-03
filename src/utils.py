# improved_utils.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.optimize import linear_sum_assignment
from scipy.stats import mode
from collections import defaultdict, Counter
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('har_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

def cosine_similarity(a, b):
    """Compute cosine similarity between two sets of vectors"""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.einsum('ik,jk->ij', a_norm, b_norm)

def visualize_alignment(embeddings1, embeddings2, labels1=None, labels2=None, filename='alignment_tsne.png'):
    """Visualize two sets of embeddings using TSNE"""
    # Combine the embeddings
    X = np.concatenate([embeddings1, embeddings2])
    if labels1 is not None and labels2 is not None:
        labels = np.concatenate([labels1, labels2])
        n1 = len(embeddings1)
        n2 = len(embeddings2)
    else:
        labels = None
    
    # Apply TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)
    
    # Plot
    plt.figure(figsize=(10, 8))
    if labels is not None:
        scatter = plt.scatter(X_tsne[:n1, 0], X_tsne[:n1, 1], c=labels[:n1], cmap='tab10', alpha=0.5, label='Set 1')
        plt.scatter(X_tsne[n1:, 0], X_tsne[n1:, 1], c=labels[n1:], cmap='tab10', marker='x', alpha=0.5, label='Set 2')
        plt.legend()
    else:
        plt.scatter(X_tsne[:n1, 0], X_tsne[:n1, 1], alpha=0.5, label='Set 1')
        plt.scatter(X_tsne[n1:, 0], X_tsne[n1:, 1], marker='x', alpha=0.5, label='Set 2')
        plt.legend()
    plt.title('TSNE Visualization of Aligned Embeddings')
    plt.savefig(filename)
    plt.close()
    return X_tsne

def load_data(dataset='UCI'):
    """Load dataset with user IDs for cross-validation"""
    logger.info(f"Loading {dataset} dataset...")
    if dataset == 'UCI':
        return load_uci()
    elif dataset == 'PAMAP2':
        return load_pamap2()
    raise ValueError(f"Unsupported dataset: {dataset}")

def load_uci(data_root='data/UCI_HAR'):
    """Load UCI HAR dataset with user IDs"""
    def read_signals(folder, set_type):
        signals = []
        sensor_axes = ['body_acc', 'body_gyro', 'total_acc']
        axes = ['x', 'y', 'z']
        for sensor in sensor_axes:
            for axis in axes:
                path = os.path.join(folder, 'Inertial Signals', f'{sensor}_{axis}_{set_type}.txt')
                signals.append(np.loadtxt(path))
        return np.concatenate(signals, axis=1)

    X_train = read_signals(f'{data_root}/train', 'train')
    X_test = read_signals(f'{data_root}/test', 'test')
    y_train = np.loadtxt(f'{data_root}/train/y_train.txt').astype(int)
    y_test = np.loadtxt(f'{data_root}/test/y_test.txt').astype(int)
    
    user_train = np.loadtxt(f'{data_root}/train/subject_train.txt').astype(int)
    user_test = np.loadtxt(f'{data_root}/test/subject_test.txt').astype(int)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    logger.info(f"UCI dataset loaded - Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test, user_train, user_test

def process_user_file(args):
    """Process a single user file in parallel"""
    file, data_root, file_to_user, window_size = args
    
    if not file.endswith('.dat'):
        return None
        
    user_id = file_to_user[file]
    segments = defaultdict(list)
    user_segments = defaultdict(list)
    
    try:
        df = pd.read_csv(os.path.join(data_root, file), delim_whitespace=True, header=None)
        df = df[df[1].between(1, 24)]
        df = df.dropna()
        
        activities = df[1].values
        activity_changes = np.where(activities[1:] != activities[:-1])[0] + 1
        activity_changes = np.concatenate(([0], activity_changes, [len(activities)]))
        
        for i in range(len(activity_changes) - 1):
            start, end = activity_changes[i], activity_changes[i+1]
            if end - start < window_size:
                continue
                
            activity = activities[start]
            segment_data = df.iloc[start:end][[2] + list(range(4, 40))].values.astype(np.float32)
            
            for j in range(0, len(segment_data) - window_size + 1, window_size // 2):
                window = segment_data[j:j+window_size]
                if len(window) == window_size:
                    segments[activity].append(window)
                    user_segments[activity].append(user_id)
    except Exception as e:
        logger.error(f"Error processing file {file}: {str(e)}")
        return None
        
    return segments, user_segments

def load_pamap2(data_root='data/PAMAP2/PAMAP2_Dataset/Protocol', window_size=192):
    """Load PAMAP2 dataset with user IDs and stratified splitting"""
    file_to_user = {}
    files = [f for f in os.listdir(data_root) if f.endswith('.dat')]
    for i, file in enumerate(sorted(files)):
        file_to_user[file] = i
    
    num_workers = max(1, min(multiprocessing.cpu_count() - 1, len(files)))
    args_list = [(file, data_root, file_to_user, window_size) for file in files]
    
    all_segments = defaultdict(list)
    all_user_segments = defaultdict(list)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_user_file, args_list))
    
    for result in results:
        if result is None:
            continue
        segments, user_segments = result
        for label, seqs in segments.items():
            all_segments[label].extend(seqs)
            all_user_segments[label].extend(user_segments[label])
    
    X, y, user_ids = [], [], []
    for label, seqs in all_segments.items():
        user_list = all_user_segments[label]
        for seq_idx, seq in enumerate(seqs):
            user_id = user_list[seq_idx]
            X.append(seq)
            y.append(label)
            user_ids.append(user_id)
    
    X = np.stack(X)
    y = np.array(y, dtype=np.int32)
    user_ids = np.array(user_ids, dtype=np.int32)
    
    unique_users = np.unique(user_ids)
    
    X_train, X_val_test, y_train, y_val_test, user_train, user_val_test = [], [], [], [], [], []
    
    for user in unique_users:
        user_mask = (user_ids == user)
        X_user = X[user_mask]
        y_user = y[user_mask]
        
        if len(np.unique(y_user)) > 1:
            X_u_train, X_u_val_test, y_u_train, y_u_val_test = train_test_split(
                X_user, y_user, test_size=0.4, stratify=y_user, random_state=42
            )
            
            class_counts = Counter(y_u_val_test)
            if min(class_counts.values()) < 2:
                stratify_arg = None
            else:
                stratify_arg = y_u_val_test

            X_u_val, X_u_test, y_u_val, y_u_test = train_test_split(
                X_u_val_test, y_u_val_test, test_size=0.5, stratify=stratify_arg, random_state=42
            )

        else:
            split_idx1 = int(0.6 * len(X_user))
            split_idx2 = int(0.8 * len(X_user))
            X_u_train, X_u_val, X_u_test = X_user[:split_idx1], X_user[split_idx1:split_idx2], X_user[split_idx2:]
            y_u_train, y_u_val, y_u_test = y_user[:split_idx1], y_user[split_idx1:split_idx2], y_user[split_idx2:]
        
        X_train.append(X_u_train)
        y_train.append(y_u_train)
        user_train.extend([user] * len(X_u_train))
        
        X_val_test.append(np.concatenate([X_u_val, X_u_test]))
        y_val_test.append(np.concatenate([y_u_val, y_u_test]))
        user_val_test.extend([user] * (len(X_u_val) + len(X_u_test)))
    
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    user_train = np.array(user_train)
    
    X_val_test = np.concatenate(X_val_test)
    y_val_test = np.concatenate(y_val_test)
    user_val_test = np.array(user_val_test)
    
    scaler = StandardScaler()
    N_train, W, D = X_train.shape
    X_train_flat = X_train.reshape(-1, D)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(N_train, W, D)
    
    N_val_test, W, D = X_val_test.shape
    X_val_test_flat = X_val_test.reshape(-1, D)
    X_val_test_scaled = scaler.transform(X_val_test_flat).reshape(N_val_test, W, D)
    
    val_mask = np.arange(len(user_val_test)) % 2 == 0
    
    X_val = X_val_test_scaled[val_mask]
    y_val = y_val_test[val_mask]
    user_val = user_val_test[val_mask]
    
    X_test = X_val_test_scaled[~val_mask]
    y_test = y_val_test[~val_mask]
    user_test = user_val_test[~val_mask]
    
    logger.info(f"PAMAP2 dataset loaded - Train: {X_train_scaled.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    logger.info(f"Users in train: {np.unique(user_train)}, Val: {np.unique(user_val)}, Test: {np.unique(user_test)}")
    
    return X_train_scaled, y_train, X_val, y_val, X_test, y_test, user_train, user_val, user_test

def sliding_window(data, window_size, step, labels=None, user_ids=None):
    """Apply sliding window with user ID tracking and confidence scores"""
    from numpy.lib.stride_tricks import sliding_window_view
    
    try:
        if len(data.shape) == 2:
            windows = sliding_window_view(data, window_shape=(window_size, data.shape[1]))[::step, 0]
            windows = np.array([w for w in windows])
        else:
            n_samples = (len(data) - window_size) // step + 1
            windows = []
            for i in range(n_samples):
                start = i * step
                end = start + window_size
                data_window = data[start:end]
                if data_window.shape[0] == window_size:
                    windows.append(data_window)
            windows = np.stack(windows) if windows else np.array([])
    except Exception as e:
        logger.warning(f"Stride tricks failed, falling back to loop implementation: {str(e)}")
        n_samples = (len(data) - window_size) // step + 1
        windows = []
        for i in range(n_samples):
            start = i * step
            end = start + window_size
            data_window = data[start:end]
            if data_window.shape[0] == window_size:
                windows.append(data_window)
        windows = np.stack(windows) if windows else np.array([])
    
    results = [windows]
    
    if labels is not None:
        if len(windows) > 0:
            try:
                label_windows = sliding_window_view(labels, window_shape=(window_size,))[::step, 0]
                weights = np.linspace(0.5, 1.0, window_size)
                majority = np.array([
                    np.bincount(window, weights=weights[:len(window)]).argmax()
                    for window in label_windows
                ])
                confidences = np.array([
                    np.max(np.bincount(window, weights=weights[:len(window)])) / np.sum(weights[:len(window)])
                    for window in label_windows
                ])
                
                # Log low confidence windows
                low_conf_mask = confidences < 0.6
                if np.any(low_conf_mask):
                    logger.warning(f"Found {low_conf_mask.sum()} windows with label confidence < 0.6")
            except Exception:
                label_windows = []
                for i in range(len(windows)):
                    start = i * step
                    end = start + window_size
                    label_windows.append(labels[start:end])
                label_windows = np.stack(label_windows)
                majority = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 1, label_windows)
                confidences = np.array([
                    np.max(np.bincount(window)) / len(window)
                    for window in label_windows
                ])
            results.append(majority)
            results.append(confidences)
        else:
            results.append(np.array([]))
            results.append(np.array([]))
    
    if user_ids is not None:
        if len(windows) > 0:
            user_windows = []
            for i in range(len(windows)):
                start = i * step
                end = start + window_size
                if np.all(user_ids[start:end] == user_ids[start]):
                    user_windows.append(user_ids[start])
                else:
                    user_windows.append(-1)
            results.append(np.array(user_windows, dtype=np.int32))
        else:
            results.append(np.array([]))
    
    return tuple(results)

class SinkhornAlign:
    """Optimal transport-based alignment with entropy regularization and class-aware normalization"""
    def __init__(self, n_iter=20, epsilon=0.1, class_aware=False):
        self.n_iter = n_iter
        self.epsilon = epsilon
        self.class_aware = class_aware

    def __call__(self, A, B, class_labels=None):
        dist = np.linalg.norm(A[:, None] - B[None, :], axis=2)
        
        if self.class_aware and class_labels is not None:
            # Class-aware normalization
            for cls in np.unique(class_labels):
                cls_mask = (class_labels == cls)
                if cls_mask.sum() > 1:
                    dist[cls_mask] /= dist[cls_mask].max(axis=1, keepdims=True)
        
        K = np.exp(-dist / self.epsilon)
        
        u = np.ones((K.shape[0],)) / K.shape[0]
        v = np.ones((K.shape[1],)) / K.shape[1]
        
        for _ in range(self.n_iter):
            u = 1.0 / (K @ v + 1e-10)
            v = 1.0 / (K.T @ u + 1e-10)
        
        P = np.diag(u) @ K @ np.diag(v)
        row_ind, col_ind = linear_sum_assignment(-P)
        return row_ind, col_ind, P

class QuantumHungarian:
    """Time-aware alignment with temporal coherence and probabilistic assignments"""
    def __init__(self, temp=0.2, n_iter=100, feature_weight=0.7, softmax_temp=0.1):
        self.temp = temp
        self.n_iter = n_iter
        self.feature_weight = feature_weight
        self.softmax_temp = softmax_temp
        self.rng = np.random.default_rng(42)

    def __call__(self, A, B, return_prob=False):
        A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
        feature_sim = np.dot(A_norm, B_norm.T)
        
        time_dist = np.abs(np.arange(A.shape[0])[:, None] - np.arange(B.shape[0])[None, :])
        time_sim = np.exp(-time_dist / self.temp)
        
        combined_sim = self.feature_weight * feature_sim + (1 - self.feature_weight) * time_sim
        
        if return_prob:
            # Convert to probability distribution
            prob_matrix = np.exp(combined_sim / self.softmax_temp)
            prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)
            return prob_matrix
        else:
            row_ind, col_ind = linear_sum_assignment(-combined_sim)
            return row_ind, col_ind, combined_sim

def evaluate_alignment(y_true, y_shuffled, row_ind, col_ind, prob_matrix=None):
    """Calculate alignment accuracy with additional metrics and probabilistic evaluation"""
    if len(row_ind) == 0:
        return 0.0, 0.0, 0.0
    
    valid_indices = [i for i in range(len(row_ind))
                     if row_ind[i] < len(y_true) and col_ind[i] < len(y_shuffled)]
    
    if not valid_indices:
        return 0.0, 0.0, 0.0
        
    correct = (y_true[row_ind[valid_indices]] == y_shuffled[col_ind[valid_indices]]).astype(int)
    accuracy = correct.mean()
    
    from sklearn.metrics import f1_score
    f1 = f1_score(y_true[row_ind[valid_indices]], y_shuffled[col_ind[valid_indices]], average='macro')
    
    temporal_coherence = np.corrcoef(row_ind[valid_indices], col_ind[valid_indices])[0, 1]
    
    # Probabilistic evaluation if available
    prob_score = 0.0
    if prob_matrix is not None:
        matched_probs = prob_matrix[row_ind[valid_indices], col_ind[valid_indices]]
        prob_score = np.mean(matched_probs)
    
    return accuracy, f1, temporal_coherence, prob_score

class EarlyStopping:
    """Early stopping handler for training loops"""
    def __init__(self, patience=10, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False
    
    def __call__(self, epoch, value):
        if (self.mode == 'min' and value < self.best_value - self.min_delta) or \
           (self.mode == 'max' and value > self.best_value + self.min_delta):
            self.best_value = value
            self.counter = 0
            self.best_epoch = epoch
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False

class EnsemblePredictor:
    """Ensemble predictor for combining multiple models"""
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights is not None else np.ones(len(models)) / len(models)
    
    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        
        if hasattr(self, 'weights') and self.weights is not None:
            weighted_votes = np.zeros((predictions.shape[1], np.max(predictions) + 1))
            for i, clf_preds in enumerate(predictions):
                for j, pred in enumerate(clf_preds):
                    weighted_votes[j, pred] += self.weights[i]
            y_pred_ensemble = np.argmax(weighted_votes, axis=1)
        else:
            y_pred_ensemble = mode(predictions, axis=0)[0].flatten()
        return y_pred_ensemble
    
    def predict_proba(self, X):
        probas = np.array([model.predict_proba(X) for model in self.models])
        return np.average(probas, axis=0, weights=self.weights)
