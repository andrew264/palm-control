import enum
import time
from typing import Optional, Tuple

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


def angle_between_vectors(v1, v2):
    """
  Calculates the angle between two vectors.

  Args:
    v1: The first vector.
    v2: The second vector.

  Returns:
    The angle between the two vectors in degrees.
  """
    dot_product = np.dot(v1, v2)
    angle = np.arccos(dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.degrees(angle)


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

    def do_action(self, disable_clicks: bool = False):
        # MOUSE MOVEMENT
        if self.is_missing:
            return
        if self.are_these_straight(fingers=INDEX_FINGER):
            # calculate the difference between the current and the last pointer location
            pointer_location = self.coordinates_of(HandLandmark.INDEX_FINGER_TIP)
            delta_x = (pointer_location[0] - self._last_pointer_location[0]) * SCREEN_WIDTH
            delta_y = (pointer_location[1] - self._last_pointer_location[1]) * SCREEN_HEIGHT
            delta_x = int(delta_x)
            delta_y = int(delta_y)
            match closest := self.get_closest_finger_to_thumb():
                case None:
                    pyautogui.moveRel(delta_x, delta_y, duration=0.01, _pause=False)
                    print('Moving mouse')
                # case HandLandmark.INDEX_FINGER_TIP:
                #     if self.allow_click():
                #         pyautogui.scroll(delta_y)
                #     print('Scrolling')
                case HandLandmark.MIDDLE_FINGER_TIP:
                    if not disable_clicks:
                        pyautogui.dragRel(delta_x, delta_y, _pause=False)
                    print('Dragging')
                case HandLandmark.RING_FINGER_TIP:
                    if self.allow_click() and not disable_clicks:
                        pyautogui.click(_pause=False)
                    print('Clicking')
                case HandLandmark.PINKY_TIP:
                    if self.allow_click() and not disable_clicks:
                        pyautogui.rightClick(_pause=False)
                    print('Right clicking')
                case _:
                    print('Unknown finger, wtf?', closest)
            self._last_pointer_location = self.coordinates_of(HandLandmark.INDEX_FINGER_TIP)

    def update(self, data: Optional[np.ndarray]):
        if data is None:
            self._last_update += 1
            if self._last_update > 30:
                self.is_missing = True
            return None
        self.is_missing = False
        self._last_update = 0

        if data.ndim > 1:
            data = data.flatten()
        self.filter.update(data)

    def coordinates_of(self, part: HandLandmark) -> np.ndarray:
        return self.filter.x.reshape(-1, self.axis_dim)[part]

    def get_dist_between(self, part1: HandLandmark, part2: HandLandmark) -> float:
        return np.linalg.norm(self.coordinates_of(part1) - self.coordinates_of(part2))

    def finger_angles(self, fingers: list[HandLandmark]) -> Tuple[float, float]:
        """
        Calculates the angles formed by the fingers.
        Each finger is represented by 4 points: the base, the first joint, the second joint and the tip.
        Returns the angles formed by the first and second joints of the fingers.

        Args:
            fingers: A list of HandLandmark representing the fingers.

        Returns:
            A tuple containing the angles formed by the first and second joints of the fingers.
        """
        p1, p2, p3, p4 = self.coordinates[fingers]
        angle_p2 = angle_between_vectors(p2 - p1, p3 - p1)
        angle_p3 = angle_between_vectors(p3 - p2, p4 - p2)
        return angle_p2, angle_p3

    def are_these_straight(self, *, fingers: list[HandLandmark], threshold: float = 25.) -> bool:
        angles = self.finger_angles(fingers)
        return all(angle < threshold for angle in angles)

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
