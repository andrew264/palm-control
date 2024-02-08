import enum
import time
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


INDEX_FINGER: list[HandLandmark] = [HandLandmark.INDEX_FINGER_TIP, HandLandmark.INDEX_FINGER_DIP,
                                    HandLandmark.INDEX_FINGER_PIP, HandLandmark.INDEX_FINGER_MCP]


def shoelace_formula(points: np.ndarray) -> float:
    """Calculate the area of a polygon using the Shoelace formula."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


class Hand:
    def __init__(self, axis_dim: int = 2, num_landmarks: int = 21, enable_smoothing: bool = False):
        size = num_landmarks * axis_dim  # 21 landmarks, 2 coordinates each
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
            self.filter.Q *= 5e-3
        else:
            self.filter.Q *= 1
        self.previous_pointer_location = (0, 0)
        self.is_missing = True
        self._last_update = 0
        self._last_pointer_location = (0, 0)
        self._last_click_event = time.time()

    def allow_click(self):
        if time.time() - self._last_click_event > 1.:
            self._last_click_event = time.time()
            return True
        return False

    @property
    def coordinates(self) -> Optional[np.ndarray]:
        if not self.is_missing:
            self.filter.predict()
            return self.filter.x.reshape(-1, self.axis_dim)
        return None

    def do_action(self):
        # MOUSE MOVEMENT
        if self.are_these_straight(fingers=INDEX_FINGER):
            print('Index finger is straight')
            # calculate the difference between the current and the last pointer location
            pointer_location = self.coordinates_of(HandLandmark.INDEX_FINGER_TIP)
            delta_x = (pointer_location[0] - self._last_pointer_location[0]) * SCREEN_WIDTH
            delta_y = (pointer_location[1] - self._last_pointer_location[1]) * SCREEN_HEIGHT
            match closest := self.get_closest_finger_to_thumb():
                case None:
                    pyautogui.moveRel(delta_x, delta_y, duration=0.01, _pause=False)
                    print('Moving mouse')
                # case HandLandmark.INDEX_FINGER_TIP:
                #     if self.allow_click():
                #         pyautogui.scroll(delta_y)
                #     print('Scrolling')
                # case HandLandmark.MIDDLE_FINGER_TIP:
                #     pyautogui.dragRel(delta_x, delta_y)
                #     print('Dragging')
                # case HandLandmark.RING_FINGER_TIP:
                #     if self.allow_click():
                #         pyautogui.click()
                #     print('Clicking')
                # case HandLandmark.PINKY_TIP:
                #     if self.allow_click():
                #         pyautogui.rightClick()
                #     print('Right clicking')
                case _:
                    print('Unknown finger, wtf?', closest)
            self._last_pointer_location = self.coordinates_of(HandLandmark.INDEX_FINGER_TIP)

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

        return self.filter.x.reshape(-1, self.axis_dim)

    def coordinates_of(self, part: HandLandmark) -> np.ndarray:
        return self.filter.x.reshape(-1, self.axis_dim)[part]

    def get_dist_between(self, part1: HandLandmark, part2: HandLandmark) -> float:
        return np.linalg.norm(self.coordinates_of(part1) - self.coordinates_of(part2))

    def are_these_straight(self, *, fingers: list[HandLandmark], threshold: float = 1e-3) -> bool:
        points = self.coordinates[fingers]
        area = shoelace_formula(points)
        return area < threshold

    def get_closest_finger_to_thumb(self, threshold: float = 7e-2) -> Optional[HandLandmark]:
        thumb = HandLandmark.THUMB_TIP
        closest = min(
            (finger for finger in HandLandmark if (finger.name.endswith('TIP') and finger != thumb)),
            key=lambda finger: self.get_dist_between(finger, thumb)
        )
        if self.get_dist_between(closest, thumb) < threshold:
            return closest
        else:
            return None

    def __repr__(self):
        return f"Hand({self.coordinates})"
