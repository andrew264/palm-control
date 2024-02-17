import os

import torch
from torch.utils.data import DataLoader

from gesture_rec.gesture_dataset import GestureDataset
from utils import load_gesture_model

choices_file = "choices.txt"
if not os.path.exists(choices_file):
    raise FileNotFoundError(f"File {choices_file} not found")

dataset_file = "dataset.jsonl"
if not os.path.exists(dataset_file):
    raise FileNotFoundError(f"File {dataset_file} not found")

model_save_path = "../models/gesture_model.pth"


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
    with open(choices_file, "r") as file:
        labels = file.read().splitlines()
        labels = [label.strip() for label in labels]
        labels = sorted(labels)

    # load dataset
    dataset = GestureDataset(file_path=dataset_file, labels=labels)

    output_size = len(labels)

    # da sequential model
    model = load_gesture_model(model_save_path, output_size)
    train_model(model, dataset, model_save_path, epochs=1000, batch_size=64)
