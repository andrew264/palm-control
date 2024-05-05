import csv
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from gesture_network import GestureNet
from utils import (get_gesture_class_labels, batch_normalize_landmarks, batch_rotate_points, batch_horizontal_flip,
                   batch_depth_flip)

MAX_ROTATION_ANGLE = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GestureDataset(Dataset):
    def __init__(self, file_path: str, _labels: list):
        self.data = []
        self.labels = _labels
        self.num_classes = len(_labels)
        label_to_idx = {label: idx for idx, label in enumerate(_labels)}
        with open(file_path, "r") as file:
            for line in file:
                self.data.append(json.loads(line))

        for i in range(len(self.data)):
            self.data[i]["landmarks"] = torch.tensor(self.data[i]["landmarks"]).reshape(-1, 3)
            self.data[i]["label"] = label_to_idx[self.data[i]["label"]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int | slice) -> tuple[torch.Tensor, int]:
        item = self.data[idx]
        return item["landmarks"], item["label"]


def train_collate_fn(batch):
    landmarks, target = zip(*batch)
    landmarks = torch.stack(landmarks)
    landmarks = batch_normalize_landmarks(landmarks)
    if torch.rand(1).item() > 0.5:
        landmarks = batch_horizontal_flip(landmarks)
    if torch.rand(1).item() > 0.5:
        landmarks = batch_depth_flip(landmarks)
    landmarks = batch_rotate_points(landmarks, MAX_ROTATION_ANGLE)
    return landmarks, torch.tensor(target)


def val_collate_fn(batch):
    landmarks, target = zip(*batch)
    landmarks = torch.stack(landmarks)
    landmarks = batch_normalize_landmarks(landmarks)
    return landmarks, torch.tensor(target)


def plot(accuracy_hist: list[float], loss_hist: list[float]):
    import matplotlib.pyplot as plt
    ig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(range(1, len(accuracy_hist) + 1), accuracy_hist, color='b', label='Accuracy')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Accuracy', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(range(1, len(loss_hist) + 1), loss_hist, color='r', label='Loss')
    ax2.set_ylabel('Loss', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='upper right')

    plt.title('Accuracy and Loss History')
    plt.tight_layout()
    # plt.savefig("./gesture_rec/accuracy_history.png")
    plt.show()


def train_model(model: torch.nn.Module, dataset: Dataset, epochs=10, batch_size=32):
    print("Training model...")
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    total_steps = epochs * len(dataloader)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.5, total_iters=total_steps)

    accuracy_hist = []
    loss_hist = []
    for epoch in range(epochs):
        accum_loss = 0
        accum_accuracy = 0
        for landmarks, target in dataloader:
            optimizer.zero_grad()

            landmarks = landmarks.to(device)
            target = target.to(device)

            outputs = model(landmarks)
            outputs = F.log_softmax(outputs, dim=1)
            loss = criterion(outputs, target)
            loss_hist.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()

            accum_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == target).sum().item() / len(predicted) * 100
            accum_accuracy += accuracy
            accuracy_hist.append(accuracy)

        if (epoch + 1) % 100 == 0:
            loss = accum_loss / len(dataloader)
            accuracy = accum_accuracy / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
                  f"lr = {scheduler.get_last_lr()[0]:.6f}")
    plot(accuracy_hist, loss_hist)


def stats(model: torch.nn.Module, dataset: Dataset, num_classes: int, labels: list[str], csv_filename: str):
    model.eval()
    total = [0] * num_classes
    wrong = [0] * num_classes
    wrong_predictions = []

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=val_collate_fn)
    with torch.no_grad(), open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'True Output', 'Model Prediction'])  # Writing header
        for (idx, (landmarks, target)) in enumerate(dataloader):
            landmarks = landmarks.to(device)
            target = target.to(device)
            outputs = model(landmarks)
            _, predicted = torch.max(outputs, 1)
            target = target.item()
            total[target] += 1
            if predicted != target:
                wrong[target] += 1
                wrong_predictions.append((idx, target, predicted.item()))

        for idx, target, prediction in wrong_predictions:
            writer.writerow([idx, labels[target], labels[prediction]])

    for i in range(num_classes):
        print(f"Accuracy for {labels[i]}: {100 * (total[i] - wrong[i]) / total[i]:.2f}%")
        print(f"Wrong predictions: {wrong[i]}/{total[i]}")


if __name__ == "__main__":
    choices_file = "./gesture_rec/choices.txt"
    if not os.path.exists(choices_file):
        raise FileNotFoundError(f"File {choices_file} not found")
    labels = get_gesture_class_labels(choices_file)

    dataset_file = "./gesture_rec/dataset.jsonl"
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"File {dataset_file} not found")

    # load dataset
    print("Loading dataset...")
    data = GestureDataset(file_path=dataset_file, _labels=labels, )
    num_classes = len(labels)

    # da model
    model_ = GestureNet(hidden_size=48, output_size=num_classes).to(device)
    print(model_)
    train_model(model_, data, epochs=2000, batch_size=512)
    stats(model_, GestureDataset(file_path=dataset_file, _labels=labels), num_classes, labels,
          csv_filename="./gesture_rec/stats.csv")

    # export model to onnx
    dummy_input = torch.randn(1, 21, 3, device=device)
    torch.onnx.export(model_, dummy_input, "./models/gesture_model.onnx", verbose=False)
    print("Model exported to ONNX")
