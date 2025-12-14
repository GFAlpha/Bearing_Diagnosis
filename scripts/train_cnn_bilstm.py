import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

# =========================
# Config
# =========================
DATA_DIR = "data/splits"
RESULT_DIR = "results/cnn_bilstm"

EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
RUNS = 5
NUM_CLASSES = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Model
# =========================
class CNN_BiLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(64 * 2, num_classes)

    def forward(self, x):
        # x: [B, 1024] or [B, 1, 1024]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.cnn(x)            # [B, 32, T]
        x = x.permute(0, 2, 1)     # [B, T, 32]

        out, _ = self.lstm(x)      # [B, T, 128]
        out = out[:, -1, :]        # last timestep

        return self.fc(out)

# =========================
# Train / Eval
# =========================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            preds.append(logits.argmax(1).cpu())
            labels.append(y)

    return accuracy_score(
        torch.cat(labels),
        torch.cat(preds)
    )

def measure_inference_time(model, loader, repeat=20):
    model.eval()
    x, _ = next(iter(loader))
    x = x.to(DEVICE)

    with torch.no_grad():
        # warm-up
        for _ in range(5):
            _ = model(x)

        torch.cuda.synchronize() if DEVICE == "cuda" else None
        start = time.time()

        for _ in range(repeat):
            _ = model(x)

        torch.cuda.synchronize() if DEVICE == "cuda" else None
        end = time.time()

    return (end - start) / repeat

# =========================
# Main
# =========================
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    # ===== Load data =====
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    X_val   = np.load(f"{DATA_DIR}/X_val.npy")
    y_val   = np.load(f"{DATA_DIR}/y_val.npy")
    X_test  = np.load(f"{DATA_DIR}/X_test.npy")
    y_test  = np.load(f"{DATA_DIR}/y_test.npy")

    test_accs = []
    train_times = []
    infer_times = []

    for run in range(RUNS):
        print(f"\n=== Run {run + 1}/{RUNS} ===")

        model = CNN_BiLSTM(NUM_CLASSES).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train).float(),
                torch.tensor(y_train).long()
            ),
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_val).float(),
                torch.tensor(y_val).long()
            ),
            batch_size=BATCH_SIZE
        )

        test_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_test).float(),
                torch.tensor(y_test).long()
            ),
            batch_size=BATCH_SIZE
        )

        # ===== Train =====
        best_val = 0.0
        best_state = None

        start_time = time.time()

        for epoch in range(EPOCHS):
            train_one_epoch(model, train_loader, criterion, optimizer)
            val_acc = evaluate(model, val_loader)

            if val_acc > best_val:
                best_val = val_acc
                best_state = model.state_dict()

        train_time = time.time() - start_time
        train_times.append(train_time)

        # ===== Test =====
        model.load_state_dict(best_state)
        test_acc = evaluate(model, test_loader)
        test_accs.append(test_acc)

        infer_time = measure_inference_time(model, test_loader)
        infer_times.append(infer_time)

        print(f"Test Acc: {test_acc:.4f}")
        print(f"Train Time: {train_time:.2f}s | Inference Time: {infer_time:.4f}s")

    # ===== Save results =====
    np.save(f"{RESULT_DIR}/test_accs.npy", np.array(test_accs))
    np.save(f"{RESULT_DIR}/train_times.npy", np.array(train_times))
    np.save(f"{RESULT_DIR}/infer_times.npy", np.array(infer_times))

    meta = {
        "model": "CNN_BiLSTM",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "runs": RUNS,
        "num_classes": NUM_CLASSES
    }
    np.save(f"{RESULT_DIR}/meta.npy", meta)

    print("\n===== Final Statistics =====")
    print("Test Accuracies:", test_accs)
    print(f"Mean Acc: {np.mean(test_accs):.4f}")
    print(f"Std  Acc: {np.std(test_accs):.4f}")
    print(f"Avg Train Time: {np.mean(train_times):.2f}s")
    print(f"Avg Inference Time: {np.mean(infer_times):.4f}s")

if __name__ == "__main__":
    main()
