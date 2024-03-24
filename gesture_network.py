import torch
import torch.nn as nn
import torch.nn.functional as F

from typin import HAND_LANDMARK_ANGLES, HAND_LANDMARK_DISTANCES

HAND_LANDMARK_DISTANCES = torch.tensor(HAND_LANDMARK_DISTANCES)
HAND_LANDMARK_ANGLES = torch.tensor(HAND_LANDMARK_ANGLES)


class GestureNet(nn.Module):
    def __init__(self, hidden_size: int, output_size: int):
        super(GestureNet, self).__init__()
        n_dim = 3
        self.coord_conv1 = nn.Conv1d(in_channels=n_dim, out_channels=hidden_size // 8,
                                     kernel_size=n_dim, stride=1, padding=1)
        self.coord_conv2 = nn.Conv1d(in_channels=hidden_size // 8, out_channels=hidden_size // 4,
                                     kernel_size=n_dim, stride=1, padding=1)
        self.coord_conv3 = nn.Conv1d(in_channels=hidden_size // 4, out_channels=hidden_size // 2,
                                     kernel_size=n_dim, stride=1, padding=1)
        self.pooling = nn.MaxPool1d(kernel_size=2)
        self.coord_norm = nn.LayerNorm(hidden_size)

        self.rad_proj = nn.Linear(len(HAND_LANDMARK_ANGLES), hidden_size)
        self.rad_norm = nn.LayerNorm(hidden_size)

        self.dist_proj = nn.Linear(len(HAND_LANDMARK_DISTANCES), hidden_size)
        self.dist_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        self.down_proj = nn.Linear(3 * hidden_size, output_size)

        self.act = F.relu

    @staticmethod
    @torch.no_grad()
    def get_rad(landmarks: torch.Tensor) -> torch.Tensor:
        """
        Returns the radians of the angles between the landmarks.
        """
        ba = landmarks[:, HAND_LANDMARK_ANGLES[:, 0]] - landmarks[:, HAND_LANDMARK_ANGLES[:, 1]]
        bc = landmarks[:, HAND_LANDMARK_ANGLES[:, 2]] - landmarks[:, HAND_LANDMARK_ANGLES[:, 1]]
        dot_product = torch.sum(ba * bc, dim=-1)
        norm_ba = torch.linalg.norm(ba, dim=-1)
        norm_bc = torch.linalg.norm(bc, dim=-1)
        radians = torch.acos(dot_product / (norm_ba * norm_bc))
        return radians

    @staticmethod
    @torch.no_grad()
    def get_dist(landmarks: torch.Tensor) -> torch.Tensor:
        """
        Returns the distances between the landmarks.
        """
        dists = torch.linalg.norm(
            landmarks[:, HAND_LANDMARK_DISTANCES[:, 0]] - landmarks[:, HAND_LANDMARK_DISTANCES[:, 1]], dim=-1)
        return dists

    def convoluted_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Some convoluted shit is going on here...
        """
        x = x.permute(0, 2, 1)
        x = self.pooling(self.act(self.coord_conv1(x)))
        x = self.pooling(self.act(self.coord_conv2(x)))
        x = self.pooling(self.act(self.coord_conv3(x)))
        return x.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rad = self.rad_norm(self.act(self.rad_proj(self.get_rad(x))))
        dist = self.dist_norm(self.act(self.dist_proj(self.get_dist(x))))
        cp = self.coord_norm(self.convoluted_forward(x))

        x = torch.cat([cp, rad, dist], dim=-1)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x
