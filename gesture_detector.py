import numpy as np
import torch

from hand import Hand
from typin import HandEvent, HandLandmark
from utils import load_gesture_model, normalize_landmarks, get_gesture_class_labels


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


INDEX_FINGER: list[HandLandmark] = [HandLandmark.INDEX_FINGER_TIP, HandLandmark.INDEX_FINGER_DIP,
                                    HandLandmark.INDEX_FINGER_PIP, HandLandmark.INDEX_FINGER_MCP]


class GestureDetector:
    def __init__(self, hand: Hand):
        self.hand = hand

    def detect(self) -> HandEvent:
        if self.is_event_drag():
            return HandEvent.MOUSE_DRAG
        elif self.is_event_click():
            return HandEvent.MOUSE_CLICK
        elif self.is_event_right_click():
            return HandEvent.MOUSE_RIGHT_CLICK
        elif self.is_event_audio_input():
            return HandEvent.AUDIO_INPUT
        elif self.is_event_move():
            return HandEvent.MOUSE_MOVE
        else:
            return HandEvent.MOUSE_NO_EVENT

    def _is_index_finger_extended(self, threshold_angle: float = 10.) -> bool:
        """
        Determines if the index finger is extended.
        """
        points = self.hand.coordinates_of(INDEX_FINGER)
        if points is None:
            return False
        p1, p2, p3, p4 = points
        angle_p2 = angle_between_vectors(p2 - p1, p3 - p1)
        angle_p3 = angle_between_vectors(p3 - p2, p4 - p2)
        return angle_p2 < threshold_angle and angle_p3 < threshold_angle

    def _distance_between(self, part1: HandLandmark, part2: HandLandmark) -> float:
        """
        Calculates the distance between two parts of the hand.
        """
        coordinate1 = self.hand.coordinates_of(part1)
        coordinate2 = self.hand.coordinates_of(part2)
        if coordinate1 is None or coordinate2 is None:
            return float('inf')
        return np.linalg.norm(coordinate1 - coordinate2)

    def _is_thumb_index_middle_touching(self, threshold: float = 5e-2) -> bool:
        """
        Determines if the thumb, index and middle fingers are touching.
        """
        distance_between_thumb_index = self._distance_between(HandLandmark.THUMB_TIP, HandLandmark.INDEX_FINGER_TIP)
        distance_between_index_middle = self._distance_between(HandLandmark.INDEX_FINGER_TIP,
                                                               HandLandmark.MIDDLE_FINGER_TIP)
        return distance_between_thumb_index < threshold and distance_between_index_middle < threshold

    def _is_thumb_middle_touching(self, threshold: float = 3e-2) -> bool:
        """
        Determines if the thumb and middle fingers are touching.
        """
        distance_between_thumb_middle = self._distance_between(HandLandmark.THUMB_TIP, HandLandmark.MIDDLE_FINGER_TIP)
        return distance_between_thumb_middle < threshold

    def _is_thumb_middle_pip_touching(self, threshold: float = 3e-2) -> bool:
        """
        Determines if the thumb and index fingers are touching at the PIP joint.
        """
        distance_between_thumb_index_pip = self._distance_between(HandLandmark.THUMB_TIP,
                                                                  HandLandmark.MIDDLE_FINGER_PIP)
        return distance_between_thumb_index_pip < threshold

    def _is_thumb_ring_touching(self, threshold: float = 3e-2) -> bool:
        """
        Determines if the thumb and ring fingers are touching.
        """
        distance_between_thumb_ring = self._distance_between(HandLandmark.THUMB_TIP, HandLandmark.RING_FINGER_TIP)
        return distance_between_thumb_ring < threshold

    def _is_thumb_pinky_touching(self, threshold: float = 5e-2) -> bool:
        """
        Determines if the thumb and pinky fingers are touching.
        """
        distance_between_thumb_pinky = self._distance_between(HandLandmark.THUMB_TIP, HandLandmark.PINKY_TIP)
        distance_between_thumb_pinky_mcp = self._distance_between(HandLandmark.THUMB_TIP, HandLandmark.PINKY_MCP)
        return distance_between_thumb_pinky < threshold or distance_between_thumb_pinky_mcp < threshold

    def is_event_move(self) -> bool:
        return self._is_index_finger_extended()

    def is_event_drag(self) -> bool:
        return self._is_thumb_index_middle_touching()

    def is_event_click(self) -> bool:
        return self._is_thumb_middle_touching() or self._is_thumb_middle_pip_touching()

    def is_event_right_click(self) -> bool:
        return self._is_thumb_ring_touching()

    def is_event_audio_input(self) -> bool:
        return self._is_thumb_pinky_touching()


class GestureDetectorProMax:
    def __init__(self, hand: Hand, model_path: str, labels_path: str):
        self.hand = hand
        self.labels = get_gesture_class_labels(labels_path)
        self.model = load_gesture_model(model_path, len(self.labels))

    def _do_inference(self) -> str:
        coordinates = self.hand.coordinates
        if coordinates is None:
            return "NONE"
        coordinates = normalize_landmarks(coordinates).flatten()
        outputs = self.model(torch.tensor(coordinates).unsqueeze(0))
        _, predicted = torch.max(outputs, 1)
        return self.labels[predicted]

    def detect(self) -> HandEvent:
        label = self._do_inference()
        match label:
            case "NONE" | "CLOSED_PALM":
                return HandEvent.MOUSE_NO_EVENT
            case "INDEX_POINTING" | "OPEN_PALM":
                return HandEvent.MOUSE_MOVE
            case "THUMB_MIDDLE_TOUCH":
                return HandEvent.MOUSE_CLICK
            case "3_FINGER_PINCH":
                return HandEvent.MOUSE_DRAG
            case "THUMB_RING_TOUCH":
                return HandEvent.MOUSE_RIGHT_CLICK
            case "THUMB_PINKY_TOUCH":
                return HandEvent.AUDIO_INPUT
            case "5_FINGER_PINCH":
                return HandEvent.MOUSE_SCROLL
            case "THUMBS_UP":
                return HandEvent.VOLUME_UP
            case "THUMBS_DOWN":
                return HandEvent.VOLUME_DOWN
            case "MIDDLE_UP":
                return HandEvent.COPY_TEXT
            case "MIDDLE_DOWN":
                return HandEvent.PASTE_TEXT
            case _:
                # print(f"Unknown label: {label}")
                return HandEvent.MOUSE_NO_EVENT
