import os

import cv2
import numpy as np
import torch
from torch import nn

from gesture_network import GestureFFN
from hand import Hand
from hand_tracking import HandTrackingThread
from utils import draw_landmarks_on_image

model_save_path = "../models/gesture_model.pth"
if not os.path.exists(model_save_path):
    raise FileNotFoundError(f"File {model_save_path} not found")

choices_file = "choices.txt"
if not os.path.exists(choices_file):
    raise FileNotFoundError(f"File {choices_file} not found")
with open(choices_file, "r") as file:
    labels = file.read().splitlines()
    labels = [label.strip() for label in labels]
    labels = sorted(labels)


def load_model(path: str) -> nn.Module:
    model = GestureFFN(input_size=21 * 3, hidden_size=256, output_size=len(labels))
    model.load_state_dict(torch.load(path))
    model.cpu()
    model.eval()
    return model


def normalize(landmarks: list | torch.Tensor):
    if landmarks is None:
        return torch.zeros(21 * 3)
    if isinstance(landmarks, list) or isinstance(landmarks, np.ndarray):
        landmarks = torch.tensor(landmarks).float()
    if landmarks.ndim > 1:
        landmarks = torch.flatten(landmarks)
    mean = torch.mean(landmarks, dim=0)
    std = torch.std(landmarks, dim=0)
    return (landmarks - mean) / std


@torch.inference_mode()
def predict_gesture(model, landmarks):
    landmarks = normalize(landmarks)
    outputs = model(landmarks.unsqueeze(0))
    _, predicted = torch.max(outputs, 1)
    return labels[predicted.item()]


if __name__ == '__main__':
    hand = Hand(enable_smoothing=True, axis_dim=3)
    tracking_thread = HandTrackingThread(hand=hand, num_hands=1, model_path='../models/hand_landmarker.task',
                                         camera_id=0, camera_width=1280, camera_height=720, camera_fps=30)
    tracking_thread.start()
    gesture_model = load_model(model_save_path)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.flip(frame, 1)
        if hand.coordinates is not None:
            image = draw_landmarks_on_image(image, hand.coordinates_2d)
            gesture = predict_gesture(gesture_model, hand.coordinates)
            cv2.putText(image, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Gesture Recognition", image)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
