import json
import os

import torch
from torch.utils.data import DataLoader, Dataset

from gesture_network import GestureFFN
from utils import normalize_landmarks, get_gesture_class_labels


class GestureDataset(Dataset):
    def __init__(self, file_path: str, labels: list):
        self.file_path = file_path
        self.data = []
        self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        with open(file_path, "r") as file:
            for line in file:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        landmarks = self.data[idx]["landmarks"]
        label = self.data[idx]["label"]
        return normalize_landmarks(torch.tensor(landmarks)), self.label_to_idx[label]


def train_model(model, dataset, save_path, epochs=10, batch_size=32):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        accum_loss = 0
        accum_accuracy = 0
        for landmarks, target in dataloader:
            optimizer.zero_grad()
            outputs = model(landmarks)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            accum_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            accum_accuracy += (predicted == target).sum().item() / len(predicted) * 100
        loss = accum_loss / len(dataloader)
        accuracy = accum_accuracy / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    # load and sort labels
    choices_file = "choices.txt"
    if not os.path.exists(choices_file):
        raise FileNotFoundError(f"File {choices_file} not found")

    dataset_file = "dataset.jsonl"
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"File {dataset_file} not found")

    model_save_path = "../models/gesture_model.pth"
    labels = get_gesture_class_labels(choices_file)

    # load dataset
    dataset = GestureDataset(file_path=dataset_file, labels=labels)
    num_classes = len(labels)

    # da model
    model = GestureFFN(input_size=21 * 3, hidden_size=256, output_size=num_classes)
    train_model(model, dataset, model_save_path, epochs=500, batch_size=32)
