import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python

from hand import Hand
from utils import draw_landmarks_on_image

hand = Hand(enable_smoothing=False, axis_dim=3)
NUM_HANDS = 1
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)


def load_model():
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    base_options = python.BaseOptions(model_asset_path='./models/hand_landmarker.task')
    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionRunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_tracking_confidence=0.3)
    detector = HandLandmarker.create_from_options(options)
    return detector


def draw_mp_landmarks_on_image(rgb_image, detection_result):
    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image


def do_tracking(show_window: bool = False):
    detector = load_model()
    frame_timestamp_ms = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        image = cv2.flip(image, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        results = detector.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 1
        if show_window:
            cv2.imshow("Hand Tracking", draw_mp_landmarks_on_image(image, results))
            if cv2.waitKey(1) == 27:
                break
        hand_landmarks = results.hand_landmarks
        if hand_landmarks:
            np_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks[0]])
            hand.update(np_landmarks)
        else:
            hand.update(None)


if __name__ == '__main__':
    print("Starting hand tracking thread")
    threading.Thread(target=do_tracking, daemon=True, name="Tracking-Thread").start()
    print("Waiting for hand tracking to start...")
    while hand.is_missing:
        time.sleep(0.5)
    print("Starting main loop")
    while True:
        t = time.time()
        time.sleep(1 / 120)
        img = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        if not hand.is_missing:
            coordinates = hand.coordinates_2d
            if coordinates is not None:
                img = draw_landmarks_on_image(img, coordinates)
            hand.do_action(disable_clicks=False)
        fps = 1 / (time.time() - t)
        cv2.putText(img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Hand Tracking', img)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
