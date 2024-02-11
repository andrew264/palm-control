from typing import Optional, List

import numpy as np
from filterpy.kalman import KalmanFilter

from typin import HandLandmark


class Hand:
    def __init__(self, axis_dim: int = 3, num_landmarks: int = 21, enable_smoothing: bool = False):
        size = num_landmarks * axis_dim  # 21 landmarks, 3 coordinates each
        self.axis_dim = axis_dim
        self.num_landmarks = num_landmarks
        self.enable_smoothing = enable_smoothing
        self.filter = KalmanFilter(dim_x=size, dim_z=size)

        self.filter.F = np.eye(size)
        self.filter.H = np.eye(size)
        self.filter.x = np.zeros((size, 1))
        self.filter.R *= 1e-1
        self.filter.P *= 1e-1
        if enable_smoothing:
            self.filter.Q *= 5e-2
        else:
            self.filter.Q *= 1
        self.is_missing = True
        self._last_update = 0

    @property
    def coordinates(self) -> Optional[np.ndarray]:
        if not self.is_missing:
            return self.filter.x.reshape(-1, self.axis_dim)
        return None

    @property
    def coordinates_2d(self) -> Optional[np.ndarray]:
        if not self.is_missing:
            return self.coordinates[:, :2]
        return None

    def update(self, data: Optional[np.ndarray]):
        if data is None:
            self._last_update += 1
            if self._last_update > 15:
                self.is_missing = True
            return
        self.is_missing = False
        self._last_update = 0

        if data.ndim > 1:
            data = data.flatten()
        self.filter.update(data)
        self.filter.predict()

    def coordinates_of(self, part: HandLandmark | List[HandLandmark]) -> Optional[np.ndarray]:
        return self.coordinates[part] if not self.is_missing else None

    def __repr__(self):
        return f"Hand({self.coordinates})"
