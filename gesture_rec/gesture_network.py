import torch
import torch.nn as nn


class GestureFFN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(GestureFFN, self).__init__()
        self.gate_proj = nn.Linear(input_size, hidden_size)
        self.up_proj = nn.Linear(input_size, hidden_size)
        self.down_proj = nn.Linear(hidden_size, output_size)
        self.act = nn.SiLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x)))
