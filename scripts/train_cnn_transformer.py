# scripts/train_cnn_transformer.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import os
import time

# =========================
# CNN + Transformer Model
# =========================
class CNNTransformer(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # -------- CNN Backbone --------
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # -------- Transformer Encoder --------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=2
        )

        # -------- Classifier --------
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [B, 1, T]
        x = self.cnn(x)          # [B, C, T']
        x = x.permute(0, 2, 1)   # [B, T', C]
        x = self.transformer(x) # [B, T', C]
        x = x.mean(dim=1)        # Global Average Pooling
        return self.classifier(x)

# =========================
# Train / Eval
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x).argmax(dim=1).cpu().numpy()
            preds.extend(out)
            labels.extend(y.numpy())
    return accuracy_score(labels, preds)


def inference_time(model, loader, device):
    model.eval()
    start = time.time()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _ = model(x)
    return time.time() - start

# =========================
# Main
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------- Load data --------
    X_train = np.load("data/splits/X_train.npy")
    y_train = np.load("data/splits/y_train.npy")
    X_val   = np.load("data/splits/X_val.npy")
    y_val   = np.load("data/splits/y_val.npy")
    X_test  = np.load("data/splits/X_test.npy")
    y_test  = np.load("data/splits/y_test.npy")

    def make_loader(X, y, shuffle):
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.long)
        return DataLoader(
            TensorDataset(X, y),
            batch_size=64,
            shuffle=shuffle
        )

    train_loader = make_loader(X_train, y_train, True)
    val_loader   = make_loader(X_val, y_val, False)
    test_loader  = make_loader(X_test, y_test, False)

    # -------- Logs --------
    NUM_RUNS = 5
    EPOCHS = 20

    test_accs = []
    train_times = []
    infer_times = []

    # =========================
    # Multi-run Training
    # =========================
    for run in range(NUM_RUNS):
        print(f"\n=== Run {run+1}/{NUM_RUNS} ===")

        model = CNNTransformer(num_classes=len(np.unique(y_train))).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        best_val = 0.0
        best_state = None

        start_train = time.time()

        for epoch in range(EPOCHS):
            train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_acc = evaluate(model, val_loader, device)

            if val_acc > best_val:
                best_val = val_acc
                best_state = model.state_dict()

        train_time = time.time() - start_train
        train_times.append(train_time)

        model.load_state_dict(best_state)

        test_acc = evaluate(model, test_loader, device)
        infer_t = inference_time(model, test_loader, device)

        test_accs.append(test_acc)
        infer_times.append(infer_t)

        print(f"Test Acc: {test_acc:.4f}")
        print(f"Train Time: {train_time:.2f}s | Inference Time: {infer_t:.4f}s")

    # =========================
    # Save results
    # =========================
    os.makedirs("results", exist_ok=True)

    test_accs = np.array(test_accs)
    train_times = np.array(train_times)
    infer_times = np.array(infer_times)

    np.save("results/cnn_transformer_test_accs.npy", test_accs)
    np.save("results/cnn_transformer_train_times.npy", train_times)
    np.save("results/cnn_transformer_infer_times.npy", infer_times)

    print("\n===== Final Statistics =====")
    print("Test Accuracies:", test_accs)
    print(f"Mean Acc: {test_accs.mean():.4f}")
    print(f"Std Acc : {test_accs.std():.4f}")
    print(f"Avg Train Time: {train_times.mean():.2f}s")
    print(f"Avg Inference Time: {infer_times.mean():.4f}s")


if __name__ == "__main__":
    main()
