import enum
from typing import Optional

import numpy as np
import pyautogui
from filterpy.kalman import KalmanFilter

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
pyautogui.FAILSAFE = False


class HandLandmark(enum.IntEnum):
    """The 21 hand landmarks."""
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


class Hand:
    def __init__(self):
        size = 21 * 2  # 21 landmarks, 2 coordinates each
        self.filter = KalmanFilter(dim_x=size, dim_z=size)
        self.filter.F = np.eye(size)
        self.filter.H = np.eye(size)
        self.filter.Q *= 1e-2
        self.filter.R *= 1e-1
        self.filter.x = np.zeros((size, 1))
        self.filter.P *= 1e-1
        self.previous_pointer_location = (0, 0)
        self.is_missing = True
        self._last_update = 0

    @property
    def coordinates(self) -> Optional[np.ndarray]:
        if not self.is_missing:
            self.filter.predict()
            return self.filter.x.reshape(-1, 2)
        return None

    def update(self, data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if data is None:
            self._last_update += 1
            if self._last_update > 30:
                self.is_missing = True
            return None
        self.is_missing = False
        self._last_update = 0
        self.filter.predict()
        if data.ndim > 1:
            data = data.flatten()
        self.filter.update(data)
        # fing = self.get_closest_finger_to_thumb()
        # if fing is not None:
        #     print("Closest to thumb: ", fing.name)
        # fing_dist_change = self.get_2d_dist_diff(self.coordinates_of(HandLandmark.INDEX_FINGER_TIP), self.previous_pointer_location)
        # THRESHOLD = 1e-2
        # if any(abs(fing_dist_change) > THRESHOLD):
        #     print(f"Index Finger Change: {fing_dist_change}")
        #     diff = SCREEN_WIDTH * fing_dist_change[0], SCREEN_HEIGHT * fing_dist_change[1]
        #     print(f"Diff: {diff}")
        #     pyautogui.moveRel(*diff, duration=0.01)
        self.previous_pointer_location = self.coordinates_of(HandLandmark.INDEX_FINGER_TIP)[:2]
        return self.filter.x.reshape(-1, 2)

    def coordinates_of(self, part: HandLandmark) -> np.ndarray:
        return self.filter.x.reshape(-1, 2)[part]

    def get_dist_diff(self, coord1: np.ndarray, coord2: np.ndarray) -> [float, float]:
        return coord1 - coord2

    def get_dist_between(self, part1: HandLandmark, part2: HandLandmark) -> float:
        return np.linalg.norm(self.coordinates_of(part1) - self.coordinates_of(part2))

    def get_closest_finger_to_thumb(self) -> Optional[HandLandmark]:
        THRESHOLD = 7e-2
        thumb = HandLandmark.THUMB_TIP
        closest = min(
            (finger for finger in HandLandmark if (finger.name.endswith('TIP') and finger != thumb)),
            key=lambda finger: self.get_dist_between(finger, thumb)
        )
        if self.get_dist_between(closest, thumb) < THRESHOLD:
            return closest
        else:
            return None

    def __repr__(self):
        return f"Hand({self.coordinates})"
