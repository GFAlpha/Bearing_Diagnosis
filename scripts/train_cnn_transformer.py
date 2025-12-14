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
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)           # [B, C, T]
        x = x.permute(0, 2, 1)    # [B, T, C]
        x = self.transformer(x)
        x = x.mean(dim=1)
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

    # Load data
    X_train = np.load("data/splits/X_train.npy")
    y_train = np.load("data/splits/y_train.npy")
    X_val   = np.load("data/splits/X_val.npy")
    y_val   = np.load("data/splits/y_val.npy")
    X_test  = np.load("data/splits/X_test.npy")
    y_test  = np.load("data/splits/y_test.npy")

    def make_loader(X, y, shuffle):
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(X, y), batch_size=64, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, True)
    val_loader   = make_loader(X_val, y_val, False)
    test_loader  = make_loader(X_test, y_test, False)

    NUM_RUNS = 5
    EPOCHS = 20

    test_accs, train_times, infer_times = [], [], []

    save_dir = "results/cnn_transformer"
    os.makedirs(save_dir, exist_ok=True)

    for run in range(NUM_RUNS):
        print(f"\n=== Run {run+1}/{NUM_RUNS} ===")

        model = CNNTransformer(num_classes=len(np.unique(y_train))).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        best_val = 0
        best_state = None

        start_train = time.time()

        for _ in range(EPOCHS):
            train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_acc = evaluate(model, val_loader, device)
            if val_acc > best_val:
                best_val = val_acc
                best_state = model.state_dict()

        train_t = time.time() - start_train
        train_times.append(train_t)

        model.load_state_dict(best_state)

        test_acc = evaluate(model, test_loader, device)
        infer_t = inference_time(model, test_loader, device)

        test_accs.append(test_acc)
        infer_times.append(infer_t)

        print(f"Test Acc: {test_acc:.4f}")
        print(f"Train Time: {train_t:.2f}s | Infer Time: {infer_t:.4f}s")

    # Save results
    np.save(f"{save_dir}/test_accs.npy", np.array(test_accs))
    np.save(f"{save_dir}/train_times.npy", np.array(train_times))
    np.save(f"{save_dir}/infer_times.npy", np.array(infer_times))

    with open(f"{save_dir}/summary.txt", "w", encoding="utf-8") as f:
        f.write("CNN + Transformer Results\n")
        f.write(f"Mean Acc: {np.mean(test_accs):.4f}\n")
        f.write(f"Std Acc : {np.std(test_accs):.4f}\n")
        f.write(f"Avg Train Time: {np.mean(train_times):.2f}s\n")
        f.write(f"Avg Infer Time: {np.mean(infer_times):.4f}s\n")

    print("\n===== Final Statistics =====")
    print("Mean Acc:", np.mean(test_accs))
    print("Std Acc :", np.std(test_accs))


if __name__ == "__main__":
    main()
