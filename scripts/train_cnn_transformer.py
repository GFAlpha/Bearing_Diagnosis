# 导入依赖
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import os

# CNN + Transformer 模型定义
class CNNTransformer(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # -------- CNN 特征提取 --------
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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # -------- 分类头 --------
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
        x = x.mean(dim=1)        # 全局平均池化
        return self.classifier(x)

# 训练 & 测试函数
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


# 主流程（多次随机训练 + 保存结果）
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 加载数据
    X_train = np.load("data/splits/X_train.npy")
    y_train = np.load("data/splits/y_train.npy")
    X_val   = np.load("data/splits/X_val.npy")
    y_val   = np.load("data/splits/y_val.npy")
    X_test  = np.load("data/splits/X_test.npy")
    y_test  = np.load("data/splits/y_test.npy")

    # 转 Tensor
    def make_loader(X, y, shuffle):
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(X, y), batch_size=64, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, True)
    val_loader   = make_loader(X_val, y_val, False)
    test_loader  = make_loader(X_test, y_test, False)

    test_accs = []

    for run in range(5):
        print(f"\n=== Run {run+1}/5 ===")

        model = CNNTransformer().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        best_val = 0
        best_state = None

        for epoch in range(20):
            train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_acc = evaluate(model, val_loader, device)

            if val_acc > best_val:
                best_val = val_acc
                best_state = model.state_dict()

        model.load_state_dict(best_state)
        test_acc = evaluate(model, test_loader, device)
        test_accs.append(test_acc)

        print(f"Test Acc: {test_acc:.4f}")

    test_accs = np.array(test_accs)
    os.makedirs("results", exist_ok=True)
    np.save("results/cnn_transformer_test_accs.npy", test_accs)

    print("\n===== Final Statistics =====")
    print("Test Accuracies:", test_accs)
    print("Mean:", test_accs.mean())
    print("Std:", test_accs.std())


if __name__ == "__main__":
    main()
