import os
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from mediapipe.tasks import python
import psutil

from gesture_detector import GestureDetector
from hand import Hand
from typin import HandEvent, HandLandmark
from utils import draw_landmarks_on_image, draw_mp_landmarks_on_image

hand = Hand(enable_smoothing=True, axis_dim=3)
NUM_HANDS = 1
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

prevX, prevY = 0, 0
smoothening = 6
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
pyautogui.FAILSAFE = False
is_mouse_dragging = False
last_click_time = time.time()


def set_high_priority():
    p = psutil.Process()
    if os.name == 'nt':
        p.nice(psutil.HIGH_PRIORITY_CLASS)
        print("Set high priority")


def allow_click():
    global last_click_time
    if time.time() - last_click_time > 1.0:
        last_click_time = time.time()
        return True
    return False


def enable_mouse_drag():
    global is_mouse_dragging
    is_mouse_dragging = True
    pyautogui.mouseDown(_pause=False)


def disable_mouse_drag():
    global is_mouse_dragging
    is_mouse_dragging = False
    pyautogui.mouseUp(_pause=False)


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


def start_tracking(show_window: bool = False):
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


def do_mouse_movement(x, y):
    global prevX, prevY
    x, y = x * CAMERA_WIDTH, y * CAMERA_HEIGHT
    x = np.interp(x, (300, CAMERA_WIDTH - 300), (0, SCREEN_WIDTH))
    y = np.interp(y, (200, CAMERA_HEIGHT - 200), (0, SCREEN_HEIGHT))

    if x > SCREEN_WIDTH:
        x = SCREEN_WIDTH
    if y > SCREEN_HEIGHT:
        y = SCREEN_HEIGHT

    prevX = prevX + (x - prevX) / smoothening
    prevY = prevY + (y - prevY) / smoothening

    pyautogui.moveTo(prevX, prevY, _pause=False)


if __name__ == '__main__':
    set_high_priority()
    print("Starting hand tracking thread")
    threading.Thread(target=start_tracking, daemon=True, name="Tracking-Thread").start()
    print("Waiting for hand tracking to start...")
    while hand.is_missing:
        time.sleep(0.5)
    gesture_detector = GestureDetector(hand)
    print("Starting main loop")
    while True:
        t = time.time()
        time.sleep(1 / 120)
        img = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        if not hand.is_missing:
            coordinates = hand.coordinates_2d
            match event := gesture_detector.detect():
                case HandEvent.MOUSE_DRAG:
                    if not is_mouse_dragging:
                        enable_mouse_drag()
                    coords = hand.coordinates_2d[HandLandmark.WRIST].tolist()
                    do_mouse_movement(*coords)
                case HandEvent.MOUSE_CLICK:
                    if is_mouse_dragging:
                        disable_mouse_drag()
                    if allow_click():
                        pyautogui.click(_pause=False)
                case HandEvent.MOUSE_RIGHT_CLICK:
                    if is_mouse_dragging:
                        disable_mouse_drag()
                    if allow_click():
                        pyautogui.rightClick(_pause=False)
                case HandEvent.AUDIO_INPUT:
                    if is_mouse_dragging:
                        disable_mouse_drag()
                case HandEvent.MOUSE_MOVE:
                    if is_mouse_dragging:
                        disable_mouse_drag()
                    coords = hand.coordinates_2d[HandLandmark.WRIST].tolist()
                    do_mouse_movement(*coords)
                case HandEvent.MOUSE_NO_EVENT:
                    if is_mouse_dragging:
                        disable_mouse_drag()

            fps = 1 / (time.time() - t)
            if coordinates is not None:
                img = draw_landmarks_on_image(img, coordinates)
        cv2.putText(img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Hand Tracking', img)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
