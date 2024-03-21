import torch
import torch.nn as nn
import torch.nn.functional as F

from typin import HAND_LANDMARK_ANGLES, HAND_LANDMARK_DISTANCES

HAND_LANDMARK_DISTANCES = torch.tensor(HAND_LANDMARK_DISTANCES)
HAND_LANDMARK_ANGLES = torch.tensor(HAND_LANDMARK_ANGLES)


class GestureFFN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(GestureFFN, self).__init__()
        self.coord_proj = nn.Linear(input_size, hidden_size)
        self.rad_proj = nn.Linear(len(HAND_LANDMARK_ANGLES), hidden_size)
        self.dist_proj = nn.Linear(len(HAND_LANDMARK_DISTANCES), hidden_size)
        self.coord_norm = nn.LayerNorm(hidden_size)
        self.rad_norm = nn.LayerNorm(hidden_size)
        self.dist_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=0.5)
        self.down_proj = nn.Linear(3 * hidden_size, output_size)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cp = self.coord_norm(self.coord_proj(x.flatten(start_dim=1)))
        rad = self.rad_norm(self.rad_proj(self.get_rad(x)))
        dist = self.dist_norm(self.dist_proj(self.get_dist(x)))
        x = torch.cat([cp, rad, dist], dim=-1)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x
