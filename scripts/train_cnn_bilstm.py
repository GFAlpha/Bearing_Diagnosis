import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

# =========================
# Config
# =========================
DATA_DIR = "data/splits"
RESULT_DIR = "results"
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
        """
        x: [B, 1024] or [B, 1, 1024]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)       # [B, 1, 1024]

        x = self.cnn(x)              # [B, 32, T]
        x = x.permute(0, 2, 1)       # [B, T, 32]

        out, _ = self.lstm(x)        # [B, T, 128]
        out = out[:, -1, :]          # last time step

        return self.fc(out)

# =========================
# Train & Eval
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

# =========================
# Main
# =========================
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    X_val   = np.load(f"{DATA_DIR}/X_val.npy")
    y_val   = np.load(f"{DATA_DIR}/y_val.npy")
    X_test  = np.load(f"{DATA_DIR}/X_test.npy")
    y_test  = np.load(f"{DATA_DIR}/y_test.npy")

    accs = []

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

        best_val = 0.0
        best_state = None

        for epoch in range(EPOCHS):
            train_one_epoch(model, train_loader, criterion, optimizer)
            val_acc = evaluate(model, val_loader)

            if val_acc > best_val:
                best_val = val_acc
                best_state = model.state_dict()

        model.load_state_dict(best_state)
        test_acc = evaluate(model, test_loader)
        accs.append(test_acc)

        print(f"Test Acc: {test_acc:.4f}")

    accs = np.array(accs)

    # ===== 保存结果 =====
    np.save(os.path.join(RESULT_DIR, "cnn_bilstm_test_accs.npy"), accs)

    print("\n===== Final Statistics =====")
    print("Test Accuracies:", accs)
    print(f"Mean: {accs.mean():.4f}")
    print(f"Std:  {accs.std():.4f}")

if __name__ == "__main__":
    main()
