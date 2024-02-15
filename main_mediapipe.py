import os
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import psutil
import pyautogui
import whisper

from gesture_detector import GestureDetector
from hand import Hand
from speech import SpeechThread
from typin import HandEvent, HandLandmark
from utils import draw_landmarks_on_image, load_model

hand = Hand(enable_smoothing=True, axis_dim=3)
NUM_HANDS = 1
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

prevX, prevY = 0, 0
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
pyautogui.FAILSAFE = False
is_mouse_dragging = False
last_click_time = time.time()

audio_model_name = "small.en"
audio_model = whisper.load_model(name=audio_model_name, device="cuda")


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
    pyautogui.mouseDown(button='left', _pause=False)


def disable_mouse_drag():
    global is_mouse_dragging
    is_mouse_dragging = False
    pyautogui.mouseUp(button='left', _pause=False)


def start_tracking():
    detector = load_model(NUM_HANDS)
    frame_timestamp_ms = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        image = cv2.flip(image, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        results = detector.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 1
        hand_landmarks = results.hand_landmarks
        if hand_landmarks:
            np_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks[0]])
            hand.update(np_landmarks)
        else:
            hand.update(None)


def do_mouse_movement(x, y):
    alpha = 0.7
    global prevX, prevY
    x, y = x * CAMERA_WIDTH, y * CAMERA_HEIGHT
    x = np.interp(x, (300, CAMERA_WIDTH - 300), (0, SCREEN_WIDTH))
    y = np.interp(y, (400, CAMERA_HEIGHT - 50), (0, SCREEN_HEIGHT))

    if x > SCREEN_WIDTH:
        x = SCREEN_WIDTH
    if y > SCREEN_HEIGHT:
        y = SCREEN_HEIGHT

    prevX = alpha * prevX + (1 - alpha) * x
    prevY = alpha * prevY + (1 - alpha) * y

    pyautogui.moveTo(prevX, prevY, duration=0.01, _pause=False)


if __name__ == '__main__':
    set_high_priority()
    print("Starting hand tracking thread")
    threading.Thread(target=start_tracking, daemon=True, name="Tracking-Thread").start()
    audio_thread = SpeechThread(model=audio_model, model_name=audio_model_name)
    print("Waiting for hand tracking to start...")
    while hand.is_missing:
        time.sleep(0.5)
    gesture_detector = GestureDetector(hand)
    print("Starting main loop")
    while True:
        t = time.time()
        img = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        if not hand.is_missing:
            coordinates = hand.coordinates_2d
            event = gesture_detector.detect()
            if event != HandEvent.MOUSE_DRAG and is_mouse_dragging:
                disable_mouse_drag()
            match event:
                case HandEvent.MOUSE_DRAG:
                    if not is_mouse_dragging:
                        enable_mouse_drag()
                    coords = hand.coordinates_2d[HandLandmark.WRIST].tolist()
                    do_mouse_movement(*coords)
                case HandEvent.MOUSE_CLICK:
                    if allow_click():
                        pyautogui.click(_pause=False)
                case HandEvent.MOUSE_RIGHT_CLICK:
                    if allow_click():
                        pyautogui.rightClick(_pause=False)
                case HandEvent.AUDIO_INPUT:
                    if audio_thread.is_running:
                        pass
                    elif not audio_thread.is_running and not audio_thread.finished:
                        audio_thread.start()
                    elif audio_thread.finished:
                        audio_thread = SpeechThread(model=audio_model, model_name=audio_model_name)
                        audio_thread.start()
                case HandEvent.MOUSE_MOVE:
                    coords = hand.coordinates_2d[HandLandmark.WRIST].tolist()
                    do_mouse_movement(*coords)
                case HandEvent.MOUSE_NO_EVENT:
                    pass

            fps = 1 / (time.time() - t)
            if coordinates is not None:
                img = draw_landmarks_on_image(img, coordinates)
        cv2.putText(img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, f'Event: {event.name}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Hand Tracking', img)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
