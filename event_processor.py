import contextlib
import multiprocessing
import os
import time
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import cv2
import numpy as np
import pyautogui

from constants import HEIGHT, WIDTH, EMPTY_FRAME, NUM_HANDS, DEFAULT_TRACKING_SMOOTHNESS, FPS, CAMERA_ID, \
    DEFAULT_MOUSE_SMOOTHNESS, DEFAULT_SHOW_WEBCAM, DEFAULT_POINTER_SOURCE
from gesture_detector import GestureDetectorProMax
from hand import Hand
from hand_tracking import HandTrackingThread
from speech import SpeechThread
from typin import HandLandmark, HandEvent, GUIEvents
from utils import draw_landmarks_on_image

pyautogui.FAILSAFE = False
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()


class EventProcessor(multiprocessing.Process):
    def __init__(self, gui_event_queue: Queue, ):
        super(EventProcessor, self).__init__()
        self.gui_event_queue = gui_event_queue
        self.tracking_image = SharedMemory("tracking_image", )
        self.show_webcam = DEFAULT_SHOW_WEBCAM

        # Mouse Control
        self.is_mouse_button_down = False
        self.last_click_time = time.time()
        self.screen_width, self.screen_height = None, None
        self.mouse_smoothness_alpha = DEFAULT_MOUSE_SMOOTHNESS
        self.prev_x, self.prev_y = None, None

        self.current_event = HandEvent.MOUSE_NO_EVENT

        self.hand = Hand(enable_smoothing=True, axis_dim=3, smoothness=DEFAULT_TRACKING_SMOOTHNESS)

        self.gesture_detector = None
        self.current_pointer_source: HandLandmark = DEFAULT_POINTER_SOURCE

        # Hand Tracking Thread
        self._last_video_frame = EMPTY_FRAME.copy()
        self._last_video_frame_time = time.time()
        self.video_frame_shared = SharedMemory(create=True, size=EMPTY_FRAME.nbytes, name="video_frame")
        self.hand_landmarks_queue = Queue(maxsize=3)
        self.tracking_thread = None

        # Audio Transcription
        self.audio_thread_communication_queue = Queue(maxsize=1)
        self.typewriter_queue = Queue(maxsize=1)
        self.audio_thread = None

    def create_threads(self):
        start = time.time()
        self.tracking_thread = HandTrackingThread(landmark_queue=self.hand_landmarks_queue,
                                                  num_hands=NUM_HANDS,
                                                  model_path='./models/hand_landmarker.task',
                                                  camera_id=CAMERA_ID,
                                                  camera_width=WIDTH, camera_height=HEIGHT, camera_fps=FPS)
        self.tracking_thread.start()
        print(f"Hand tracking thread started in {time.time() - start:.2f} seconds")

        start = time.time()
        self.audio_thread = SpeechThread(signal_queue=self.audio_thread_communication_queue,
                                         typewriter_queue=self.typewriter_queue)
        self.audio_thread.start()
        print(f"Speech thread started in {time.time() - start:.2f} seconds")

    def load_gesture_detector(self):
        self.gesture_detector = GestureDetectorProMax(self.hand, model_path='./models/gesture_model.onnx',
                                                      labels_path='./gesture_rec/choices.txt')

    def terminate(self):
        if self.tracking_thread is not None:
            print("Terminating tracking thread")
            self.tracking_thread.terminate()
        if self.audio_thread is not None:
            print("Terminating audio thread")
            self.audio_thread.terminate()
        self.video_frame_shared.close()
        self.video_frame_shared.unlink()
        print("Terminating event processor")
        super(EventProcessor, self).terminate()

    def update_tracking_frame(self):
        if self.show_webcam:
            frame = np.ndarray((HEIGHT, WIDTH, 3), dtype=np.uint8, buffer=self.video_frame_shared.buf)
            self._last_video_frame = frame.copy()
        else:
            self._last_video_frame = EMPTY_FRAME.copy()
        new_time = time.time()
        fps = 1 / (new_time - self._last_video_frame_time)
        self._last_video_frame_time = new_time

        if self.hand.coordinates_2d is not None:
            frame = draw_landmarks_on_image(self._last_video_frame, self.hand.coordinates_2d)
            cv2.putText(frame, f"Event: {self.current_event.name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)
            if fps is not None:
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2)
            self.tracking_image.buf[:frame.nbytes] = frame.tobytes()
        else:
            self.tracking_image.buf[:self._last_video_frame.nbytes] = self._last_video_frame.tobytes()

    def do_mouse_movement(self, x: Optional[float], y: Optional[float]):
        if x is None or y is None:
            # Reset previous coordinates if x or y is None
            self.prev_x, self.prev_y = None, None
            return

        if self.prev_x is None or self.prev_y is None:
            # Initialize previous coordinates if not already set
            self.prev_x, self.prev_y = x, y
            return

        # Smooth the mouse movement
        alpha = self.mouse_smoothness_alpha
        x_smoothed = self.prev_x * (1 - alpha) + x * alpha
        y_smoothed = self.prev_y * (1 - alpha) + y * alpha

        distance = ((x_smoothed - self.prev_x) ** 2 + (y_smoothed - self.prev_y) ** 2) ** .5
        if distance < 1e-3:
            return
        multiplier = max(distance * 25, 1.)

        dx = (x_smoothed - self.prev_x) * multiplier
        dy = (y_smoothed - self.prev_y) * multiplier

        # Calculate new coordinates
        current_x, current_y = pyautogui.position()
        new_x = current_x + dx * SCREEN_WIDTH
        new_y = current_y + dy * SCREEN_HEIGHT

        # Update previous coordinates
        self.prev_x, self.prev_y = x, y

        pyautogui.moveTo(int(new_x), int(new_y), _pause=False)

    def pinch_scroll(self, x: Optional[float], y: Optional[float]):
        if x is None or y is None:
            # Reset previous coordinates if x or y is None
            self.prev_x, self.prev_y = None, None
            return

        if self.prev_x is None or self.prev_y is None:
            # Initialize previous coordinates if not already set
            self.prev_x, self.prev_y = x, y
            return

        # Calculate the change in coordinates
        y_delta = y - self.prev_y
        x_delta = x - self.prev_x

        # Check if movement is significant
        if abs(y_delta) < 1e-3 and abs(x_delta) < 1e-3:
            return

        # Determine the scaling factor based on the operating system
        if os.name == "nt":
            y_scale = 1e4 if abs(y_delta) > abs(x_delta) else 5e4
        else:
            y_scale = 3e1 if abs(y_delta) > abs(x_delta) else 5e2

        # Apply shift key modifier if scrolling horizontally
        with pyautogui.hold("shift") if abs(y_delta) <= abs(x_delta) else contextlib.suppress():
            # Perform the scroll action
            pyautogui.scroll(int(y_delta * y_scale), _pause=False)

        self.prev_x, self.prev_y = x, y

    def allow_click(self):
        if time.time() - self.last_click_time > 0.5:
            self.last_click_time = time.time()
            return True
        return False

    def enable_mouse_drag(self):
        if not self.is_mouse_button_down:
            self.is_mouse_button_down = True
            pyautogui.mouseDown(button='left', _pause=False)

    def disable_mouse_drag(self):
        if self.is_mouse_button_down:
            self.is_mouse_button_down = False
            pyautogui.mouseUp(button='left', _pause=False)

    def do_lmb_click(self):
        if self.allow_click():
            pyautogui.leftClick(_pause=False)

    def do_rmb_click(self):
        if self.allow_click():
            pyautogui.rightClick(_pause=False)

    def do_copy_text(self):
        if self.allow_click():
            pyautogui.hotkey("ctrl", "c")

    def do_paste_text(self):
        if self.allow_click():
            pyautogui.hotkey("ctrl", "v")

    def handle_events(self):
        while not self.gui_event_queue.empty():
            item = self.gui_event_queue.get()
            if isinstance(item, tuple):
                event, value = item
            else:
                event, value = item, None
            match event:
                case GUIEvents.EXIT:
                    self.terminate()
                case GUIEvents.SHOW_WEBCAM:
                    assert isinstance(value, bool)
                    self.show_webcam = value
                case GUIEvents.TRACKING_SMOOTHNESS:
                    assert isinstance(value, float)
                    self.hand.set_filterQ(value)
                case GUIEvents.MOUSE_SMOOTHNESS:
                    assert isinstance(value, float)
                    self.mouse_smoothness_alpha = value
                case GUIEvents.MOUSE_POINTER:
                    if isinstance(value, str):
                        value = HandLandmark[value]
                    assert isinstance(value, HandLandmark)
                    self.current_pointer_source = value
                case _:
                    pass

    def do_typing(self):
        while not self.typewriter_queue.empty():
            pyautogui.write(self.typewriter_queue.get(), _pause=False)  # Write the text to the active window

    def update_hand_landmarks(self):
        while not self.hand_landmarks_queue.empty():
            self.hand.update(self.hand_landmarks_queue.get(block=False))  # Update the hand landmarks from the queue

    def run(self):
        try:
            self.create_threads()
            self.load_gesture_detector()
        except AssertionError as e:
            print(e)
        while True:
            self.handle_events()
            self.update_tracking_frame()
            self.do_typing()
            self.update_hand_landmarks()

            if not self.hand.is_missing:
                hand_coords = self.hand.coordinates_of(self.current_pointer_source)
                if hand_coords is not None:
                    x, y, _ = hand_coords.tolist()
                else:
                    x, y = None, None

                # Detect the current event
                self.current_event = self.gesture_detector.detect()

                if self.current_event != HandEvent.MOUSE_DRAG and self.is_mouse_button_down:
                    self.disable_mouse_drag()
                match self.current_event:
                    case HandEvent.MOUSE_DRAG:
                        self.enable_mouse_drag()
                        self.do_mouse_movement(x, y)
                    case HandEvent.MOUSE_CLICK:
                        self.do_lmb_click()
                    case HandEvent.MOUSE_RIGHT_CLICK:
                        self.do_rmb_click()
                    case HandEvent.AUDIO_INPUT:
                        if self.audio_thread_communication_queue.empty():
                            self.audio_thread_communication_queue.put_nowait(True)
                    case HandEvent.MOUSE_MOVE:
                        self.do_mouse_movement(x, y)
                    case HandEvent.MOUSE_SCROLL:
                        self.pinch_scroll(x, y)
                    case HandEvent.VOLUME_UP:
                        pyautogui.press("volumeup", _pause=False)
                    case HandEvent.VOLUME_DOWN:
                        pyautogui.press("volumedown", _pause=False)
                    case HandEvent.COPY_TEXT:
                        self.do_copy_text()
                    case HandEvent.PASTE_TEXT:
                        self.do_paste_text()
                    case _:
                        self.prev_x, self.prev_y = None, None
            else:
                self.current_event = HandEvent.MOUSE_NO_EVENT
                self.disable_mouse_drag()
                self.prev_x, self.prev_y = None, None
