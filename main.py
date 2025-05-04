import argparse
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch

from csi.csi_dataset import CSIDataset
from csi.csi_loader import CsiDataLoader
from csi.csi_model import DopplerRSSIFusionModel
from csi.csi_processor import CsiProcessor
from utils.algo_utils import doppler_collate_fn


def train(model, device, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for doppler_list, rssi_list, labels in train_loader:
        # Move labels to device
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(doppler_list, rssi_list)  # (B, num_classes)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * labels.size(0)
        predictions = outputs.argmax(dim=1)
        running_correct += (predictions == labels).sum().item()
        running_total += labels.size(0)

    epoch_loss = running_loss / running_total
    epoch_accuracy = running_correct / running_total

    return epoch_loss, epoch_accuracy


def evaluate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for doppler_list, rssi_list, labels in val_loader:
            labels = labels.to(device)

            outputs = model(doppler_list, rssi_list)  # (B, num_classes)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    epoch_loss = val_loss / val_total
    epoch_acc = val_correct / val_total

    return epoch_loss, epoch_acc


def train_and_evaluate():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    csi_loader = CsiDataLoader({})
    file_data = csi_loader.read_records_from_folder("data")

    csi_processor = CsiProcessor(file_data)
    all_spectrograms, all_labels, all_rssi_stats = csi_processor.process()
    train_spectrograms, validation_spectrograms, train_labels, validation_labels, train_rssi, validation_rssi = train_test_split(
        all_spectrograms, all_labels, all_rssi_stats, test_size=0.3, random_state=42)

    train_dataset = CSIDataset(train_spectrograms, train_labels, train_rssi)
    validation_dataset = CSIDataset(
        validation_spectrograms, validation_labels, validation_rssi)
    # 数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, collate_fn=doppler_collate_fn)
    val_loader = DataLoader(validation_dataset, batch_size=16,
                            shuffle=False, collate_fn=doppler_collate_fn)

    model = DopplerRSSIFusionModel(task_type='classification',
                                   doppler_in_channels=2,
                                   out_freq_dim=128,
                                   time_dim=4,
                                   doppler_hidden_size=64,
                                   rssi_max_dim=4,
                                   rssi_proj_dim=8,
                                   rssi_hidden_size=32,
                                   num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=5e-3, weight_decay=1e-3)

    # 定义学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True)

    # 定义早停参数
    patience = 50
    best_val_acc = 0.5
    epochs_no_improve = 0
    best_model_path = 'best_model.pth'

    # 训练循环
    num_epochs = 300
    for epoch in range(1, num_epochs + 1):
        # 训练
        train_loss, train_accuracy = train(
            model, device, train_loader, criterion, optimizer)

        # 验证
        validation_loss, validation_accuracy = evaluate(model, device, val_loader, criterion)

        # 更新学习率调度器
        scheduler.step(validation_accuracy)

        # 记录最佳模型
        if validation_accuracy > best_val_acc:
            best_val_acc = validation_accuracy
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> 新的最佳验证准确率: {best_val_acc:.4f}, 保存模型。")
        else:
            epochs_no_improve += 1

        # 检查早停条件
        if epochs_no_improve >= patience:
            print(f"早停触发在第 {epoch} 个 epoch。")
            break

        # 打印日志
        print(f"Epoch {epoch}/{num_epochs} | "
              f"Train Loss={train_loss:.4f}, Acc={train_accuracy:.4f} || "
              f"Val Loss={validation_loss:.4f}, Acc={validation_accuracy:.4f}")

    print("训练完成。")


def main():
    parser = argparse.ArgumentParser(
        description="Process task type and config file.")
    parser.add_argument("--task", type=str, required=True,
                        help="Type of the task to perform.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration JSON file.")
    args = parser.parse_args()

    task_type = args.task
    config_path = args.config

    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    print(f"Task Type: {task_type}")
    print(f"Config: {config["name"]}")


if __name__ == "__main__":
    main()
