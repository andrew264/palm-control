import multiprocessing

import cv2
import mediapipe as mp
import numpy as np

from utils import load_mediapipe_model


class HandTrackingThread(multiprocessing.Process):
    def __init__(self,
                 landmark_queue: multiprocessing.Queue,
                 frame_queue: multiprocessing.Queue,
                 num_hands: int,
                 model_path: str,
                 camera_id: int,
                 camera_width: int,
                 camera_height: int,
                 camera_fps: int) -> None:
        super().__init__()
        self.landmark_queue = landmark_queue
        self.frame_queue = frame_queue
        self.num_hands = num_hands
        self.model_path = model_path
        self.detector = None
        self.camera_id = camera_id
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps

    def run(self):
        self.detector = load_mediapipe_model(num_hands=self.num_hands, model_path=self.model_path)
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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if not self.frame_queue.full():
                self.frame_queue.put_nowait(image.copy())
            results = self.detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=image), counter)
            counter += 1
            landmarks = None
            if hand_landmarks := results.hand_landmarks:
                landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks[0]])
            if not self.landmark_queue.full():
                self.landmark_queue.put_nowait(landmarks)
        cap.release()
