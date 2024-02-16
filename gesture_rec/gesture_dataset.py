import json

import torch
from torch.utils.data import Dataset


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

    @staticmethod
    def normalize(landmarks: list | torch.Tensor):
        if isinstance(landmarks, list):
            landmarks = torch.tensor(landmarks)
        mean = torch.mean(landmarks, dim=0)
        std = torch.std(landmarks, dim=0)
        return (landmarks - mean) / std

    def __getitem__(self, idx):
        landmarks = self.data[idx]["landmarks"]
        label = self.data[idx]["choice"]
        return self.normalize(landmarks), self.label_to_idx[label]
