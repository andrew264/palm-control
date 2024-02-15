import threading

import cv2
import mediapipe as mp
import numpy as np

from hand import Hand
from utils import load_model


class HandTrackingThread(threading.Thread):
    def __init__(self,
                 hand: Hand,
                 num_hands: int,
                 model_path: str,
                 camera_id: int,
                 camera_width: int,
                 camera_height: int,
                 camera_fps: int) -> None:
        threading.Thread.__init__(self, daemon=True, name="HandTrackingThread")
        self.hand = hand
        self.detector = load_model(num_hands=num_hands, model_path=model_path)
        self.camera_id = camera_id
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
        counter = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.flip(frame, 1)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect_for_video(mp_image, counter)
            counter += 1
            if hand_landmarks := results.hand_landmarks:
                self.hand.update(
                    np.array([[landmark.x, landmark.y, landmark.z]
                              for landmark in hand_landmarks[0]])
                )
            else:
                self.hand.update(None)
        cap.release()
