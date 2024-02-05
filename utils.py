import enum

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


def _landmarks_list_to_array(landmark_list) -> np.ndarray:
    return np.array([[point.x, point.y] for point in landmark_list])


def _rescale_landmarks(landmarks: np.ndarray, image_shape) -> list:
    rows, cols, _ = image_shape
    return [[int(point[0] * cols), int(point[1] * rows)] for point in landmarks]


class HandLandmarkKalmanFilter:
    def __init__(self, initial_state):
        self.kf = KalmanFilter(dim_x=42, dim_z=21 * 2)
        self.kf.F = np.eye(42)
        self.kf.H = np.eye(21 * 2)
        self.kf.Q *= 8e-2
        self.kf.R *= 1e-1
        self.kf.x = np.reshape(initial_state, (42, 1))
        self.kf.P *= 1e-1

    def update(self, measurement: np.ndarray) -> np.ndarray:
        self.kf.predict()
        if measurement.ndim > 1:
            measurement = measurement.flatten()
        self.kf.update(measurement)
        return self.kf.x.reshape(-1, 2)
