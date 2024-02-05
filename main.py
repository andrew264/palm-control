import os

import cv2
import mediapipe as mp
import requests
from mediapipe.tasks import python

from utils import HAND_CONNECTIONS, _landmarks_list_to_array

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def get_hand_landmark_model():
    path = 'models/hand_landmarker.task'
    if not os.path.exists(path):
        url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
        with open(path, "wb") as f:
            print(f'Downloading model from {url}')
            f.write(requests.get(url).content)
    return path


def draw_landmarks_on_image(image, landmark_point: list):
    if not landmark_point:
        return image

    landmark_point = _landmarks_list_to_array(landmark_point, image.shape)

    colors = [(0, 0, 0), (255, 255, 255)]

    def draw_line(start, end, thickness):
        cv2.line(image, tuple(start), tuple(end), colors[0], thickness + 4)
        cv2.line(image, tuple(start), tuple(end), colors[1], thickness)

    def draw_circle(center, radius, thickness):
        cv2.circle(image, tuple(center), radius + 3, colors[0], -1)
        cv2.circle(image, tuple(center), radius, colors[1], thickness)

    for start, end in HAND_CONNECTIONS:
        draw_line(landmark_point[start], landmark_point[end], 6)

    for index in range(len(landmark_point)):
        draw_circle(landmark_point[index], 5 if index < 13 else 8, 1)

    return image


if __name__ == '__main__':
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=get_hand_landmark_model()),
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
        running_mode=VisionRunningMode.VIDEO, )
    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        timestamp = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
            image = cv2.flip(image, 1)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            hand_landmarker_result = landmarker.detect_for_video(mp_image, timestamp)
            timestamp += 1
            for hand_landmarks in hand_landmarker_result.hand_landmarks:
                image = draw_landmarks_on_image(image, hand_landmarks)
            cv2.imshow('MediaPipe Hand Landmarks', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
