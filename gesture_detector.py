import torch

from hand import Hand
from typin import HandEvent
from utils import load_gesture_model, normalize_landmarks, get_gesture_class_labels


class GestureDetectorProMax:
    def __init__(self, hand: Hand, model_path: str, labels_path: str):
        self.hand = hand
        self.labels = get_gesture_class_labels(labels_path)
        self.model = load_gesture_model(model_path, len(self.labels))

    def _do_inference(self) -> str:
        coordinates = self.hand.coordinates
        if coordinates is None:
            return "NONE"
        coordinates = normalize_landmarks(coordinates)
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
