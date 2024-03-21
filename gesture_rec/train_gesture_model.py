import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from gesture_network import GestureFFN
from utils import normalize_landmarks, get_gesture_class_labels, random_rotate_points


class GestureDataset(Dataset):
    def __init__(self, file_path: str, _labels: list, apply_random_rotation: bool = False):
        self.file_path = file_path
        self.data = []
        self.label_to_idx = {label: idx for idx, label in enumerate(_labels)}
        self.labels = _labels
        self.num_classes = len(_labels)
        self.apply_random_rotation = apply_random_rotation
        with open(file_path, "r") as file:
            for line in file:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        landmarks = np.float32(self.data[idx]["landmarks"]).reshape(-1, 3)
        label = self.data[idx]["label"]
        if self.apply_random_rotation:
            landmarks = random_rotate_points(landmarks)
        return normalize_landmarks(landmarks), self.label_to_idx[label]


def train_model(model: GestureFFN, dataset: GestureDataset, save_path: str, epochs=10, batch_size=32):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        accum_loss = 0
        accum_accuracy = 0
        for landmarks, target in dataloader:
            optimizer.zero_grad()
            outputs = model(landmarks)
            outputs = F.softmax(outputs, dim=1)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            accum_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            accum_accuracy += (predicted == target).sum().item() / len(predicted) * 100
        if (epoch + 1) % 25 == 0:
            loss = accum_loss / len(dataloader)
            accuracy = accum_accuracy / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")


def stats(model: GestureFFN, dataset: GestureDataset):
    correct = [0] * dataset.num_classes
    total = [0] * dataset.num_classes
    wrong = [0] * dataset.num_classes
    with torch.no_grad():
        for landmarks, target in dataset:
            outputs = model(torch.tensor(landmarks).unsqueeze(0))
            _, predicted = torch.max(outputs, 1)
            total[target] += 1
            if predicted == target:
                correct[target] += 1
            else:
                wrong[target] += 1
    for i in range(dataset.num_classes):
        print(f"Accuracy of {dataset.labels[i]}: {correct[i] / total[i] * 100:.2f}%")
        print(f"Wrong predictions: {wrong[i]}/{total[i]}")


if __name__ == "__main__":
    choices_file = "./gesture_rec/choices.txt"
    if not os.path.exists(choices_file):
        raise FileNotFoundError(f"File {choices_file} not found")
    labels = get_gesture_class_labels(choices_file)

    dataset_file = "./gesture_rec/dataset.jsonl"
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"File {dataset_file} not found")

    model_save_path = "./models/gesture_model.pth"

    # load dataset
    data = GestureDataset(file_path=dataset_file, _labels=labels, apply_random_rotation=True)
    num_classes = len(labels)

    # da model
    model_ = GestureFFN(input_size=21 * 3, hidden_size=128, output_size=num_classes)
    model_.train()
    train_model(model_, data, model_save_path, epochs=1000, batch_size=128)
    model_.eval()
    stats(model_, GestureDataset(file_path=dataset_file, _labels=labels))

    # export model to onnx
    dummy_input = torch.randn(1, 21, 3)
    torch.onnx.export(model_, dummy_input, "./models/gesture_model.onnx", verbose=False)
    print("Model exported to ONNX")
