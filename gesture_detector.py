import torch

from hand import Hand
from typin import HandEvent
from utils import get_gesture_class_labels, load_onnx_model, run_inference_on_onnx_model

MAPPING = {
    "NONE": HandEvent.MOUSE_NO_EVENT,
    "CLOSED_PALM": HandEvent.MOUSE_NO_EVENT,
    "INDEX_POINTING": HandEvent.MOUSE_MOVE,
    "OPEN_PALM": HandEvent.MOUSE_MOVE,
    "THUMB_MIDDLE_TOUCH": HandEvent.MOUSE_CLICK,
    "3_FINGER_PINCH": HandEvent.MOUSE_DRAG,
    "THUMB_RING_TOUCH": HandEvent.MOUSE_RIGHT_CLICK,
    "THUMB_PINKY_TOUCH": HandEvent.AUDIO_INPUT,
    "5_FINGER_PINCH": HandEvent.MOUSE_SCROLL,
    "THUMBS_UP": HandEvent.VOLUME_UP,
    "THUMBS_DOWN": HandEvent.VOLUME_DOWN,
    "MIDDLE_UP": HandEvent.COPY_TEXT,
    "MIDDLE_DOWN": HandEvent.PASTE_TEXT,
}


class GestureDetectorProMax:
    def __init__(self, hand: Hand, model_path: str, labels_path: str):
        self.hand = hand
        self.labels = get_gesture_class_labels(labels_path)
        self.model = load_onnx_model(model_path)

    def _do_inference(self, coordinates) -> str:
        outputs = run_inference_on_onnx_model(self.model, coordinates)
        predicted = torch.argmax(torch.tensor(outputs), dim=1).item()
        return self.labels[predicted]

    def detect(self) -> HandEvent:
        coordinates = self.hand.coordinates
        if coordinates is None:
            return HandEvent.MOUSE_NO_EVENT
        return MAPPING[self._do_inference(coordinates)]
