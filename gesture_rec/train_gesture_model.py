import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from gesture_network import GestureFFN
from utils import get_gesture_class_labels, batch_normalize_landmarks, batch_rotate_points

MAX_ROTATION_ANGLE = 30
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
    landmarks = batch_rotate_points(landmarks, MAX_ROTATION_ANGLE)
    landmarks = batch_normalize_landmarks(landmarks)
    return landmarks, torch.tensor(target)


def val_collate_fn(batch):
    landmarks, target = zip(*batch)
    landmarks = torch.stack(landmarks)
    landmarks = batch_normalize_landmarks(landmarks)
    return landmarks, torch.tensor(target)


def train_model(model: GestureFFN, dataset: GestureDataset, epochs=10, batch_size=32):
    print("Training model...")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collate_fn)
    for epoch in range(epochs):
        accum_loss = 0
        accum_accuracy = 0
        for landmarks, target in dataloader:
            optimizer.zero_grad()

            landmarks = landmarks.to(device)
            target = target.to(device)

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
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=val_collate_fn)
    with torch.no_grad():
        for landmarks, target in dataloader:
            landmarks = landmarks.to(device)
            target = target.to(device)
            outputs = model(landmarks)
            _, predicted = torch.max(outputs, 1)
            target = target.item()
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
    print("Loading dataset...")
    data = GestureDataset(file_path=dataset_file, _labels=labels, )
    num_classes = len(labels)

    # da model
    model_ = GestureFFN(hidden_size=128, output_size=num_classes)
    print(model_)
    model_.train()
    model_.to(device, )
    train_model(model_, data, epochs=1000, batch_size=128)
    model_.eval()
    stats(model_, GestureDataset(file_path=dataset_file, _labels=labels))

    # export model to onnx
    dummy_input = torch.randn(1, 21, 3, device=device)
    torch.onnx.export(model_, dummy_input, "./models/gesture_model.onnx", verbose=False)
    print("Model exported to ONNX")
