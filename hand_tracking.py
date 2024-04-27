import multiprocessing
import os
from multiprocessing.shared_memory import SharedMemory

import cv2
import mediapipe as mp
import numpy as np

from constants import HEIGHT, WIDTH, FPS, CAMERA_ID, NUM_HANDS
from utils import load_mediapipe_model


class HandTrackingThread(multiprocessing.Process):
    def __init__(self, landmark_queue: multiprocessing.Queue, video_frame_name: str) -> None:
        super().__init__()
        self.landmark_queue = landmark_queue
        self.video_frame = SharedMemory(video_frame_name)
        self.detector = None

    def run(self):
        print(f"{self.__class__.__name__}'s PID: {os.getpid()}")
        self.detector = load_mediapipe_model(num_hands=NUM_HANDS, model_path='./models/hand_landmarker.task')
        cap = cv2.VideoCapture(CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        counter = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.flip(frame, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.video_frame.buf[:image.nbytes] = image.tobytes()
            # image = cv2.resize(image, (720, 480), interpolation=cv2.INTER_NEAREST)  # Resize for faster processing ig
            results = self.detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=image), counter)
            counter += 1
            landmarks = None
            if hand_landmarks := results.hand_landmarks:
                landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks[0]])
            if not self.landmark_queue.full():
                self.landmark_queue.put_nowait(landmarks)
        cap.release()
