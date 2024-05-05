import multiprocessing
import os
import time
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Tuple

import cv2
import numpy as np
import pyautogui

from constants import HEIGHT, WIDTH, EMPTY_FRAME, DEFAULT_TRACKING_SMOOTHNESS, DEFAULT_MOUSE_SMOOTHNESS, \
    DEFAULT_SHOW_WEBCAM, DEFAULT_POINTER_SOURCE
from gesture_detector import GestureDetectorProMax
from hand import Hand
from hand_tracking import HandTrackingThread
from speech import SpeechThread
from typin import HandLandmark, HandEvent, GUIEvents
from utils import draw_landmarks_on_image, get_volume_linux, adjust_volume_linux

pyautogui.FAILSAFE = False
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()


class EventProcessor(multiprocessing.Process):
    _iteration_delay = 1 / 60

    def __init__(self, gui_event_queue: Queue, tracking_image_name: str):
        super().__init__()
        self.gui_event_queue = gui_event_queue
        self.tracking_image = SharedMemory(tracking_image_name)
        self.show_webcam = DEFAULT_SHOW_WEBCAM

        # Mouse Control
        self.is_mouse_button_down = False
        self.last_click_time = time.time()
        self.screen_width, self.screen_height = None, None
        self.mouse_smoothness_alpha = DEFAULT_MOUSE_SMOOTHNESS
        self.prev_coords = None

        self.current_event = HandEvent.MOUSE_NO_EVENT

        self.hand = Hand(enable_smoothing=True, axis_dim=3, smoothness=DEFAULT_TRACKING_SMOOTHNESS)

        self.gesture_detector = None
        self.current_pointer_source: HandLandmark = DEFAULT_POINTER_SOURCE

        # Hand Tracking Thread
        self._last_video_frame_time = time.time()
        self.video_frame_shared = SharedMemory(create=True, size=EMPTY_FRAME.nbytes)
        self.hand_landmarks_queue = Queue(maxsize=3)
        self.tracking_thread = None

        # Audio Transcription
        self.audio_thread_communication_queue = Queue(maxsize=1)
        self.typewriter_queue = Queue(maxsize=1)
        self.audio_thread = None

    def initialize_threads(self):
        start = time.time()
        self.tracking_thread = HandTrackingThread(landmark_queue=self.hand_landmarks_queue,
                                                  video_frame_name=self.video_frame_shared.name)
        self.tracking_thread.start()
        print(f"Hand tracking thread started in {time.time() - start:.2f} seconds")

        start = time.time()
        self.audio_thread = SpeechThread(signal_queue=self.audio_thread_communication_queue,
                                         typewriter_queue=self.typewriter_queue)
        self.audio_thread.start()
        print(f"Speech thread started in {time.time() - start:.2f} seconds")

        self.gesture_detector = GestureDetectorProMax(self.hand,
                                                      model_path='./models/gesture_model.onnx',
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
        exit(0)

    @property
    def pointer_coordinates(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        # coordinates of previous and current frame
        if self.hand.coordinates is None:
            self.prev_coords = None
            return None
        curr_coords = self.hand.coordinates_of(self.current_pointer_source)
        if self.prev_coords is None:
            self.prev_coords = curr_coords
            return None
        if np.array_equal(curr_coords, self.prev_coords):
            return None
        out = (self.prev_coords, curr_coords)
        self.prev_coords = curr_coords
        return out

    def update_tracking_frame(self):
        if self.show_webcam:
            frame = np.ndarray((HEIGHT, WIDTH, 3), dtype=np.uint8, buffer=self.video_frame_shared.buf)
            _last_video_frame = frame.copy()
        else:
            _last_video_frame = EMPTY_FRAME.copy()
        new_time = time.time()
        fps = 1 / (new_time - self._last_video_frame_time)
        self._last_video_frame_time = new_time

        if self.hand.coordinates_2d is not None:
            frame = draw_landmarks_on_image(_last_video_frame, self.hand.coordinates_2d)
            cv2.putText(frame, f"Event: {self.current_event.name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)
            if fps is not None:
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2)
            self.tracking_image.buf[:frame.nbytes] = frame.tobytes()
        else:
            self.tracking_image.buf[:_last_video_frame.nbytes] = _last_video_frame.tobytes()

    def smooth_coords(self, curr_coord: float, prev_coord: float) -> float:
        return prev_coord * (1 - self.mouse_smoothness_alpha) + curr_coord * self.mouse_smoothness_alpha

    def do_mouse_movement(self, ):
        coords = self.pointer_coordinates
        if coords is None:
            return
        prev_coords, current_coords = coords
        prev_x, prev_y, prev_z = prev_coords
        current_x, current_y, current_z = current_coords
        depth_mul = np.interp(-current_z, (0.01, 0.2), (60, 70))

        # Smooth the mouse movement
        x_smoothed = self.smooth_coords(current_x, prev_x)
        y_smoothed = self.smooth_coords(current_y, prev_y)

        distance = ((x_smoothed - prev_x) ** 2 + (y_smoothed - prev_y) ** 2) ** .5
        if distance < 1e-3:
            return
        multiplier = max(distance * depth_mul, 1.)

        dx = (x_smoothed - prev_x) * multiplier
        dy = (y_smoothed - prev_y) * multiplier

        # Calculate new coordinates
        current_x, current_y = pyautogui.position()
        new_x = current_x + dx * SCREEN_WIDTH
        new_y = current_y + dy * SCREEN_HEIGHT

        if 0 <= new_x <= SCREEN_WIDTH and 0 <= new_y <= SCREEN_HEIGHT:
            pyautogui.moveTo(int(new_x), int(new_y), _pause=False)

    def pinch_scroll(self):
        coords = self.pointer_coordinates
        if coords is None:
            return
        prev_coords, current_coords = coords
        prev_x, prev_y, prev_z = prev_coords
        current_x, current_y, current_z = current_coords

        # Smooth the mouse movement
        x_smoothed = self.smooth_coords(current_x, prev_x)
        y_smoothed = self.smooth_coords(current_y, prev_y)

        distance = ((x_smoothed - prev_x) ** 2 + (y_smoothed - prev_y) ** 2) ** .5
        if distance < 1e-3:
            return

        dx: float = (x_smoothed - prev_x) * 100
        dy: float = (y_smoothed - prev_y) * 100

        if os.name == "nt":
            if abs(dx) > abs(dy):
                dx = np.interp(dx, (-5, 5), (-1000, 1000)).item()
                with pyautogui.hold('shift', _pause=False):
                    pyautogui.scroll(int(dx), _pause=False)
            else:
                dy = np.interp(dy, (-5, 5), (-1000, 1000)).item()
                pyautogui.scroll(int(dy), _pause=False)
        else:
            if abs(dx) > abs(dy):  # horizontal scroll
                pyautogui.hscroll(-dx, _pause=False)
            else:  # vertical scroll
                dy = np.interp(dy, (-5, 5), (-2.5, 2.5)).item()  # vertical scroll is funky without this
                pyautogui.vscroll(dy, _pause=False)

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

    def increase_volume(self):
        if self.allow_click():
            if os.name == "nt":
                pyautogui.press("volumeup", _pause=False)
            else:
                if get_volume_linux() < 100:
                    adjust_volume_linux(5)

    def decrease_volume(self):
        if self.allow_click():
            if os.name == "nt":
                pyautogui.press("volumedown", _pause=False)
            else:
                if get_volume_linux() > 0:
                    adjust_volume_linux(-5)

    def handle_gui_events(self):
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
            pyautogui.write(self.typewriter_queue.get_nowait(), _pause=False)  # Write the text to the active window

    def update_hand_landmarks(self):
        while not self.hand_landmarks_queue.empty():
            self.hand.update(self.hand_landmarks_queue.get())  # Update the hand landmarks from the queue

    def run(self):
        print(f"{self.__class__.__name__}'s PID: {os.getpid()}")
        try:
            self.initialize_threads()
        except AssertionError as e:
            print(e)
            self.terminate()
        start_time = time.time()
        try:
            while True:
                self.handle_gui_events()
                self.update_hand_landmarks()
                self.update_tracking_frame()
                self.do_typing()

                if not self.hand.is_missing:
                    # Detect the current event
                    self.current_event = self.gesture_detector.detect()

                    if self.current_event != HandEvent.MOUSE_DRAG and self.is_mouse_button_down:
                        self.disable_mouse_drag()
                    match self.current_event:
                        case HandEvent.MOUSE_DRAG:
                            self.enable_mouse_drag()
                            self.do_mouse_movement()
                        case HandEvent.MOUSE_CLICK:
                            self.do_lmb_click()
                            self.do_mouse_movement()
                        case HandEvent.MOUSE_RIGHT_CLICK:
                            self.do_rmb_click()
                        case HandEvent.AUDIO_INPUT:
                            if self.audio_thread_communication_queue.empty():
                                self.audio_thread_communication_queue.put_nowait(True)
                        case HandEvent.MOUSE_MOVE:
                            self.do_mouse_movement()
                        case HandEvent.MOUSE_SCROLL:
                            self.pinch_scroll()
                        case HandEvent.VOLUME_UP:
                            self.increase_volume()
                        case HandEvent.VOLUME_DOWN:
                            self.decrease_volume()
                        case HandEvent.COPY_TEXT:
                            self.do_copy_text()
                        case HandEvent.PASTE_TEXT:
                            self.do_paste_text()
                        case _:
                            self.prev_coords = None
                else:
                    self.current_event = HandEvent.MOUSE_NO_EVENT
                    self.disable_mouse_drag()

                elapsed_time = time.time() - start_time
                remaining_time = max(self._iteration_delay - elapsed_time, 0)
                time.sleep(remaining_time)
                start_time = time.time()
        except KeyboardInterrupt:
            self.terminate()
