import time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
import pyautogui
import whisper
from PIL import Image, ImageTk

from gesture_detector import GestureDetector
from hand import Hand
from hand_tracking import HandTrackingThread
from speech import SpeechThread
from typin import HandEvent, HandLandmark
from utils import draw_landmarks_on_image

WIDTH, HEIGHT = 1280, 720
FPS = 30

hand = Hand(enable_smoothing=True, axis_dim=3)
NUM_HANDS = 1
EMPTY_FRAME = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
DEFAULT_TRACKING_SMOOTHNESS: float = 5e-2
DEFAULT_MOUSE_SMOOTHNESS: float = 0.7
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
pyautogui.FAILSAFE = False

tracking_thread = HandTrackingThread(hand=hand, num_hands=NUM_HANDS, model_path='./models/hand_landmarker.task',
                                     camera_id=0, camera_width=WIDTH, camera_height=HEIGHT, camera_fps=FPS)
audio_model_name = "small.en"
audio_model = whisper.load_model(name=audio_model_name, device="cpu")


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Palm Control GUI")
        self.root.geometry(f"{WIDTH}x{HEIGHT + 100}")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.resizable(False, False)
        self.root.bind("<Escape>", lambda e: self.on_close())
        self.root.config(bg="black")

        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")

        self.tracking_image_label = None
        self.controls_frame = None
        self.tracking_smoothness_label = None
        self.tracking_smoothness = None
        self.mouse_smoothness_label = None
        self.mouse_smoothness = None

        self.last_click_time = time.time()
        self.is_mouse_button_down = False
        self.mouse_smoothness_alpha = DEFAULT_MOUSE_SMOOTHNESS
        self.prev_x, self.prev_y = 0, 0
        self.current_event = HandEvent.MOUSE_NO_EVENT

        self.audio_thread = SpeechThread(model=audio_model,
                                         model_name=audio_model_name)
        self.gesture_detector = GestureDetector(hand)

        self.create_widgets()
        self.update_frame()
        self.process_loop()

    def on_close(self):
        print("Closing the application")
        self.root.destroy()

    def create_widgets(self):
        font = ("Roboto Mono", 12)
        self.tracking_image_label = ttk.Label(self.root)
        self.tracking_image_label.pack()

        self.controls_frame = ttk.Frame(self.root, padding=10)
        self.controls_frame.pack()

        # Tracking Smoothness Section
        tracking_frame = ttk.Frame(self.controls_frame)
        tracking_frame.pack(fill="x", pady=(0, 10))

        self.tracking_smoothness_label = ttk.Label(tracking_frame, text="Tracking Smoothness:", font=font)
        self.tracking_smoothness_label.pack(side="left", padx=(0, 5))

        self.tracking_smoothness = ttk.Scale(tracking_frame, from_=1., to=5e-3,
                                             orient="horizontal", length=200)
        self.tracking_smoothness.set(DEFAULT_TRACKING_SMOOTHNESS)
        self.tracking_smoothness.config(command=self.update_tracking_smoothness)
        self.tracking_smoothness.pack(fill="x")

        # Mouse Smoothness Section
        mouse_frame = ttk.Frame(self.controls_frame)
        mouse_frame.pack(fill="x")

        self.mouse_smoothness_label = ttk.Label(mouse_frame, text="Mouse Smoothness:", font=font)
        self.mouse_smoothness_label.pack(side="left", padx=(0, 5))

        self.mouse_smoothness = ttk.Scale(mouse_frame, from_=0, to=1,
                                          orient="horizontal", length=200)
        self.mouse_smoothness.set(DEFAULT_MOUSE_SMOOTHNESS)
        self.mouse_smoothness.config(command=self.update_mouse_smoothness)
        self.mouse_smoothness.pack(fill="x")

    def get_tracking_frame(self) -> np.ndarray:
        frame = EMPTY_FRAME.copy()
        if hand.coordinates_2d is not None:
            frame = draw_landmarks_on_image(frame, hand.coordinates_2d)
        cv2.putText(frame, f"Event: {self.current_event.name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
        return frame

    @staticmethod
    def update_tracking_smoothness(value):
        hand.set_filterQ(float(value))

    def update_mouse_smoothness(self, value):
        self.mouse_smoothness_alpha = float(value)

    def update_frame(self):
        frame = self.get_tracking_frame()
        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=img)
        self.tracking_image_label.config(image=img)
        self.tracking_image_label.image = img
        self.root.after(1000 // FPS, self.update_frame)

    def run(self):
        self.root.mainloop()

    def do_mouse_movement(self, x, y):
        x, y = x * WIDTH, y * HEIGHT
        x = np.interp(x, (300, WIDTH - 300), (0, SCREEN_WIDTH))
        y = np.interp(y, (400, HEIGHT - 50), (0, SCREEN_HEIGHT))

        if x > SCREEN_WIDTH:
            x = SCREEN_WIDTH
        if y > SCREEN_HEIGHT:
            y = SCREEN_HEIGHT

        self.prev_x = self.mouse_smoothness_alpha * self.prev_x + (1 - self.mouse_smoothness_alpha) * x
        self.prev_y = self.mouse_smoothness_alpha * self.prev_y + (1 - self.mouse_smoothness_alpha) * y

        pyautogui.moveTo(self.prev_x, self.prev_y, duration=.0, _pause=False)

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
        if not hand.is_missing:
            self.current_event = self.gesture_detector.detect()
            if self.current_event != HandEvent.MOUSE_DRAG and self.is_mouse_button_down:
                self.disable_mouse_drag()
            match self.current_event:
                case HandEvent.MOUSE_DRAG:
                    self.enable_mouse_drag()
                    self.do_mouse_movement(*hand.coordinates_2d[HandLandmark.WRIST].tolist())
                case HandEvent.MOUSE_CLICK:
                    if self.allow_click():
                        pyautogui.click(_pause=False)
                case HandEvent.MOUSE_RIGHT_CLICK:
                    if self.allow_click():
                        pyautogui.rightClick(_pause=False)
                case HandEvent.AUDIO_INPUT:
                    if self.audio_thread.is_running:
                        pass
                    elif not self.audio_thread.is_running and not self.audio_thread.finished:
                        self.audio_thread.start()
                    elif self.audio_thread.finished:
                        self.audio_thread = SpeechThread(model=audio_model, model_name=audio_model_name)
                        self.audio_thread.start()
                case HandEvent.MOUSE_MOVE:
                    self.do_mouse_movement(*hand.coordinates_2d[HandLandmark.WRIST].tolist())
                case _:
                    pass
        else:
            self.current_event = HandEvent.MOUSE_NO_EVENT
            self.disable_mouse_drag()
        self.root.after(8, self.process_loop)


if __name__ == '__main__':
    app = GUI()
    tracking_thread.start()
    app.run()
