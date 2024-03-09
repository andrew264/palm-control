import os
import time
import tkinter as tk
from multiprocessing import Queue
from tkinter import ttk
from typing import Optional

import cv2
import numpy as np
import pyautogui
import sv_ttk
from PIL import Image, ImageTk

from gesture_detector import GestureDetector, GestureDetectorProMax  # noqa
from hand import Hand
from hand_tracking import HandTrackingThread
from speech import SpeechThread
from typin import HandEvent, HandLandmark
from utils import draw_landmarks_on_image

WIDTH, HEIGHT = 1280, 720
FPS = 60

NUM_HANDS = 1
EMPTY_FRAME = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
DEFAULT_TRACKING_SMOOTHNESS: float = 5e-1
DEFAULT_MOUSE_SMOOTHNESS: float = 0.7
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
pyautogui.FAILSAFE = False


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Palm Control GUI")
        self.root.geometry(f"{WIDTH}x{HEIGHT + 125}")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.resizable(False, False)
        self.root.bind("<Escape>", lambda e: self.on_close())
        self.root.bind("<Control-q>", lambda e: self.on_close())
        self.root.config(bg="black")

        sv_ttk.set_theme("dark", self.root)
        self.style = ttk.Style(self.root)
        # self.style.theme_use("clam")
        font = ("Roboto Mono", 12)
        padding = 10
        self.style.configure("TButton", font=font)
        self.style.configure("TLabel", font=font, padding=(padding, padding, 0, 0))
        self.style.configure("TScale", font=font, padding=(padding, padding, 0, 0))
        self.style.configure("TCheckbutton", font=font, padding=(padding, padding, 0, 0))
        self.style.configure("TCombobox", font=font, padding=(padding, padding, 0, 0))

        # Widgets
        self.tracking_image_label = None
        self.controls_frame = None
        self.tracking_smoothness_label = None
        self.tracking_smoothness = None
        self.show_webcam_var = None
        self.show_webcam_checkbox = None
        self.mouse_smoothness_label = None
        self.mouse_smoothness = None
        self.mouse_pointer_label = None
        self.mouse_pointer_source = None
        self.mouse_pointer_dropdown = None

        # Hand Tracking Thread
        self._last_video_frame = EMPTY_FRAME.copy()
        self.hand = Hand(enable_smoothing=True, axis_dim=3, smoothness=DEFAULT_TRACKING_SMOOTHNESS)
        self.video_frame_queue = Queue(maxsize=30)
        self.hand_landmarks_queue = Queue(maxsize=30)
        self.tracking_thread = HandTrackingThread(landmark_queue=self.hand_landmarks_queue,
                                                  frame_queue=self.video_frame_queue,
                                                  num_hands=NUM_HANDS,
                                                  model_path='./models/hand_landmarker.task',
                                                  camera_id=0,
                                                  camera_width=WIDTH, camera_height=HEIGHT, camera_fps=FPS)
        self.tracking_thread.start()

        # Mouse control
        self.last_click_time = time.time()
        self.is_mouse_button_down = False
        self.mouse_smoothness_alpha = DEFAULT_MOUSE_SMOOTHNESS
        self.prev_x, self.prev_y = None, None
        self.current_event = HandEvent.MOUSE_NO_EVENT

        # Audio Transcription
        self.audio_thread_communication_queue = Queue(maxsize=1)
        self.typewriter_queue = Queue(maxsize=1)
        self.audio_thread = SpeechThread(signal_queue=self.audio_thread_communication_queue,
                                         typewriter_queue=self.typewriter_queue)
        self.audio_thread.start()  # Start the thread to avoid latency when the user starts speaking

        # Gesture Detection
        # self.gesture_detector = GestureDetector(hand)
        self.gesture_detector = GestureDetectorProMax(self.hand, model_path='./models/gesture_model.pth',
                                                      labels_path='./gesture_rec/choices.txt'
                                                      )
        self.current_pointer_source: HandLandmark = HandLandmark.WRIST

        self.create_widgets()
        self.update_frame()
        self.process_loop()

    def on_close(self):
        print("Closing the application")
        self.audio_thread.terminate()
        self.tracking_thread.terminate()
        self.root.destroy()

    def create_widgets(self):
        self.tracking_image_label = ttk.Label(self.root)
        self.tracking_image_label.pack()

        self.controls_frame = ttk.Frame(self.root, padding=10)
        self.controls_frame.pack()

        tracking_frame = ttk.Frame(self.controls_frame)
        tracking_frame.pack(fill="x", pady=(0, 10))

        self.tracking_smoothness_label = ttk.Label(tracking_frame, text="Tracking Smoothness:", style="TLabel")
        self.tracking_smoothness_label.pack(side="left", padx=(0, 5), )

        self.tracking_smoothness = ttk.Scale(tracking_frame, from_=1., to=1e-2,
                                             orient="horizontal", length=300, style="TScale")
        self.tracking_smoothness.set(DEFAULT_TRACKING_SMOOTHNESS)
        self.tracking_smoothness.config(command=self.update_tracking_smoothness)
        self.tracking_smoothness.pack(fill="x", side="left", )

        self.show_webcam_var = tk.IntVar(value=0)
        self.show_webcam_checkbox = ttk.Checkbutton(tracking_frame, text="Show Webcam", variable=self.show_webcam_var,
                                                    style="TCheckbutton")
        self.show_webcam_checkbox.pack(side="left", padx=40)

        mouse_frame = ttk.Frame(self.controls_frame)
        mouse_frame.pack(fill="x")

        self.mouse_smoothness_label = ttk.Label(mouse_frame, text="Mouse Smoothness:", style="TLabel")
        self.mouse_smoothness_label.pack(side="left", )

        self.mouse_smoothness = ttk.Scale(mouse_frame, from_=0, to=1,
                                          orient="horizontal", length=300)
        self.mouse_smoothness.set(DEFAULT_MOUSE_SMOOTHNESS)
        self.mouse_smoothness.config(command=self.update_mouse_smoothness)
        self.mouse_smoothness.pack(fill="x", side="left", )

        self.mouse_pointer_label = ttk.Label(mouse_frame, text="Mouse Pointer Source:", style="TLabel")
        self.mouse_pointer_label.pack(side="left", )

        MOUSE_POINTER_CHOICES = [HandLandmark.WRIST.name, HandLandmark.INDEX_FINGER_TIP.name]
        self.mouse_pointer_source = tk.StringVar()
        self.mouse_pointer_source.set(MOUSE_POINTER_CHOICES[0])

        self.mouse_pointer_dropdown = ttk.Combobox(mouse_frame, state="readonly",
                                                   textvariable=self.mouse_pointer_source,
                                                   values=MOUSE_POINTER_CHOICES, style="TCombobox")
        self.mouse_pointer_dropdown.bind("<<ComboboxSelected>>", self.update_mouse_pointer)
        self.mouse_pointer_dropdown.pack(side="left", )

    def get_tracking_frame(self) -> np.ndarray:
        if self.show_webcam_var.get() == 1:
            if not self.video_frame_queue.empty():
                self._last_video_frame = self.video_frame_queue.get(block=False)
        else:
            self._last_video_frame = EMPTY_FRAME.copy()
            while not self.video_frame_queue.empty():
                self.video_frame_queue.get(block=False)
        if self.hand.coordinates_2d is not None:
            frame = draw_landmarks_on_image(self._last_video_frame, self.hand.coordinates_2d)
            cv2.putText(frame, f"Event: {self.current_event.name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)
            return frame
        else:
            return self._last_video_frame

    def update_tracking_smoothness(self, value):
        self.hand.set_filterQ(float(value))

    def update_mouse_smoothness(self, value):
        self.mouse_smoothness_alpha = float(value)

    def update_mouse_pointer(self, event):
        self.current_pointer_source = HandLandmark[self.mouse_pointer_source.get()]

    def update_frame(self):
        frame = self.get_tracking_frame()
        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=img)
        self.tracking_image_label.config(image=img)
        self.tracking_image_label.image = img
        self.root.after(8, self.update_frame)

    def run(self):
        self.root.mainloop()

    def do_mouse_movement(self, x: Optional[float], y: Optional[float]):
        if not (x and y):
            self.prev_x, self.prev_y = None, None
            return

        if self.prev_x is None or self.prev_y is None:
            self.prev_x, self.prev_y = x, y
            return

        # Smooth the mouse movement
        alpha = self.mouse_smoothness_alpha
        x = self.prev_x * (1 - alpha) + x * alpha
        y = self.prev_y * (1 - alpha) + y * alpha

        distance = ((x - self.prev_x) ** 2 + (y - self.prev_y) ** 2) ** .5
        if distance < 1e-3:
            return
        multiplier = max(distance * 25, 1.)

        dx = (x - self.prev_x) * multiplier
        dy = (y - self.prev_y) * multiplier

        # Calculate the new coordinates
        A1, B1 = pyautogui.position()
        A2 = (SCREEN_WIDTH * dx) + A1
        B2 = (SCREEN_HEIGHT * dy) + B1

        self.prev_x, self.prev_y = x, y

        pyautogui.moveTo(int(A2), int(B2), _pause=False)

    def pinch_scroll(self, x: Optional[float], y: Optional[float]):
        if not (x and y):
            self.prev_x, self.prev_y = None, None
            return

        if self.prev_x is None or self.prev_y is None:
            self.prev_x, self.prev_y = x, y
            return

        y_dist = y - self.prev_y
        x_dist = x - self.prev_x
        if abs(y_dist) < 1e-3 and abs(x_dist) < 1e-3:
            return
        if abs(y_dist) > abs(x_dist):
            if os.name == "nt":
                scale = 1e4
            else:
                scale = 3e1
            pyautogui.scroll(int(y_dist * scale), _pause=False)
        else:
            if os.name == "nt":
                scale = 5e4
            else:
                scale = 5e2
            with pyautogui.hold("shift"):
                pyautogui.scroll(int(y_dist * scale), _pause=False)
        self.prev_x, self.prev_y = x, y

    def allow_click(self):
        if time.time() - self.last_click_time > 1.0:
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

    def process_loop(self):
        while not self.typewriter_queue.empty():
            pyautogui.write(self.typewriter_queue.get(), _pause=False)  # Keyboard input happens here

        while not self.hand_landmarks_queue.empty():
            self.hand.update(self.hand_landmarks_queue.get(block=False))  # Update the hand landmarks from the queue

        if not self.hand.is_missing:
            hand_coords = self.hand.coordinates_of(self.current_pointer_source)
            if hand_coords is not None:
                x, y, _ = hand_coords.tolist()
            else:
                x, y = None, None
            self.current_event = self.gesture_detector.detect()
            if self.current_event != HandEvent.MOUSE_DRAG and self.is_mouse_button_down:
                self.disable_mouse_drag()
            match self.current_event:
                case HandEvent.MOUSE_DRAG:
                    self.enable_mouse_drag()
                    self.do_mouse_movement(x, y)
                case HandEvent.MOUSE_CLICK:
                    if self.allow_click():
                        pyautogui.click(_pause=False)
                case HandEvent.MOUSE_RIGHT_CLICK:
                    if self.allow_click():
                        pyautogui.rightClick(_pause=False)
                case HandEvent.AUDIO_INPUT:
                    if self.audio_thread_communication_queue.empty():
                        self.audio_thread_communication_queue.put(True)
                    self.prev_x, self.prev_y = None, None
                case HandEvent.MOUSE_MOVE:
                    self.do_mouse_movement(x, y)
                case HandEvent.MOUSE_SCROLL:
                    self.pinch_scroll(x, y)
                case HandEvent.VOLUME_UP:
                    pyautogui.press("volumeup", _pause=False)
                case HandEvent.VOLUME_DOWN:
                    pyautogui.press("volumedown", _pause=False)
                case HandEvent.COPY_TEXT:
                    if self.allow_click():
                        pyautogui.hotkey("ctrl", "c", )
                case HandEvent.PASTE_TEXT:
                    if self.allow_click():
                        pyautogui.hotkey("ctrl", "v", )
                case _:
                    self.prev_x, self.prev_y = None, None
        else:
            self.current_event = HandEvent.MOUSE_NO_EVENT
            self.disable_mouse_drag()
            self.prev_x, self.prev_y = None, None
        self.root.after(4, self.process_loop)


if __name__ == '__main__':
    app = GUI()
    app.run()
