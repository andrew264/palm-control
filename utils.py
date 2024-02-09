import cv2
import numpy as np
from filterpy.kalman import KalmanFilter

HAND_PALM_CONNECTIONS = ((0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17))

HAND_THUMB_CONNECTIONS = ((1, 2), (2, 3), (3, 4))

HAND_INDEX_FINGER_CONNECTIONS = ((5, 6), (6, 7), (7, 8))

HAND_MIDDLE_FINGER_CONNECTIONS = ((9, 10), (10, 11), (11, 12))

HAND_RING_FINGER_CONNECTIONS = ((13, 14), (14, 15), (15, 16))

HAND_PINKY_FINGER_CONNECTIONS = ((17, 18), (18, 19), (19, 20))

HAND_CONNECTIONS = frozenset().union(*[
    HAND_PALM_CONNECTIONS, HAND_THUMB_CONNECTIONS,
    HAND_INDEX_FINGER_CONNECTIONS, HAND_MIDDLE_FINGER_CONNECTIONS,
    HAND_RING_FINGER_CONNECTIONS, HAND_PINKY_FINGER_CONNECTIONS
])


def _landmarks_list_to_array(landmark_list) -> np.ndarray:
    return np.array([[point.x, point.y, point.z] for point in landmark_list])


def _rescale_landmarks(landmarks: np.ndarray, height: int, width: int) -> list:
    return [[int(point[0] * height), int(point[1] * width)] for point in landmarks]


class KalmanFilterObj:
    def __init__(self, size: int, enable_smoothing: bool = False):
        self.enable_smoothing = enable_smoothing
        self.filter = KalmanFilter(dim_x=size, dim_z=size)
        self.filter.F = np.eye(size)
        self.filter.H = np.eye(size)
        self.filter.x = np.zeros(size)
        self.filter.P = np.eye(size) * 1000
        self.filter.R *= 1e-1
        self.filter.Q *= 5e-2
        self.is_missing = True
        self._last_update = 0

    def update(self, data: np.ndarray) -> np.ndarray:
        if data is None:
            self._last_update += 1
            if self._last_update > 10:
                self.is_missing = True
        self.is_missing = False
        if self.enable_smoothing:
            self.filter.update(data)
            return self.filter.x
        return data

    def predict(self) -> np.ndarray:
        if self.enable_smoothing:
            self.filter.predict()
            return self.filter.x
        return self.filter.x


def draw_landmarks_on_image(_img: np.ndarray, _points: np.ndarray):
    _points = _rescale_landmarks(_points, *_img.shape[:2])
    colors = [(0, 0, 0), (255, 255, 255)]

    def draw_line(_start, _end, thickness):
        cv2.line(_img, tuple(_start), tuple(_end), colors[0], thickness + 4)
        cv2.line(_img, tuple(_start), tuple(_end), colors[1], thickness)

    def draw_circle(center, radius, thickness):
        cv2.circle(_img, tuple(center), radius + 3, colors[0], -1)
        cv2.circle(_img, tuple(center), radius, colors[1], thickness)

    for start, end in HAND_CONNECTIONS:
        draw_line(_points[start], _points[end], 6)

    for index in range(len(_points)):
        draw_circle(_points[index], 5, 1)

    return _img
