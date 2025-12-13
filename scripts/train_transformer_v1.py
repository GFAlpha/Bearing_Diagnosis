import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ======================
# Config
# ======================
DATA_DIR = "data/splits"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

NUM_RUNS = 5
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Dataset
# ======================
def load_dataset():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    def to_loader(X, y, shuffle=False):
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # [B, L, 1]
        y = torch.tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=shuffle)

    return (
        to_loader(X_train, y_train, True),
        to_loader(X_val, y_val, False),
        to_loader(X_test, y_test, False),
    )

# ======================
# Positional Encoding
# ======================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

# ======================
# Transformer Model
# ======================
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, NUM_CLASSES)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = x.transpose(1, 2)      # [B, d_model, L]
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# ======================
# Train / Eval
# ======================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# ======================
# Main
# ======================
def main():
    train_loader, val_loader, test_loader = load_dataset()
    test_accs = []

    for run in range(NUM_RUNS):
        print(f"\n=== Run {run+1}/{NUM_RUNS} ===")

        model = TransformerClassifier().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        best_val = 0
        best_state = None

        for epoch in range(EPOCHS):
            train_epoch(model, train_loader, criterion, optimizer)
            val_acc = evaluate(model, val_loader)

            if val_acc > best_val:
                best_val = val_acc
                best_state = model.state_dict()

        model.load_state_dict(best_state)
        test_acc = evaluate(model, test_loader)
        test_accs.append(test_acc)
        print(f"Test Acc: {test_acc:.4f}")

    test_accs = np.array(test_accs)
    np.save(os.path.join(RESULT_DIR, "transformer_test_accs.npy"), test_accs)

    print("\n===== Final Statistics =====")
    print("Test Accuracies:", test_accs)
    print("Mean:", test_accs.mean())
    print("Std:", test_accs.std())

if __name__ == "__main__":
    main()
