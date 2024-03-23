import torch

from hand import Hand
from typin import HandEvent
from utils import get_gesture_class_labels, load_onnx_model, run_inference_on_onnx_model


class GestureDetectorProMax:
    def __init__(self, hand: Hand, model_path: str, labels_path: str):
        self.hand = hand
        self.labels = get_gesture_class_labels(labels_path)
        self.model = load_onnx_model(model_path)

    def _do_inference(self) -> str:
        coordinates = self.hand.coordinates
        if coordinates is None:
            return "NONE"
        outputs = run_inference_on_onnx_model(self.model, coordinates)
        predicted = torch.argmax(torch.tensor(outputs), dim=1).item()
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
