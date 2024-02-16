import json
import os
import sys
import tkinter as tk
from tkinter.font import Font
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, '../')

from utils import load_model, draw_landmarks_on_image

cap = cv2.VideoCapture(0)
WIDTH, HEIGHT = 1280, 720
FPS = 30
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

detector = load_model(num_hands=1, model_path='../models/hand_landmarker.task')
dataset_json = []
dataset_json_path = "dataset.jsonl"
if os.path.exists(dataset_json_path):
    with open(dataset_json_path, "r") as f:
        dataset_json = [json.loads(line) for line in f]


def get_landmarks(frame, frame_count) -> Optional[np.ndarray]:
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    results = detector.detect_for_video(mp_frame, frame_count)
    hand_landmarks = results.hand_landmarks
    if hand_landmarks:
        np_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks[0]])
        return np_landmarks
    return None


class VideoGUI:
    def __init__(self, master: tk.Tk, choices: List[str]):
        self.master = master
        self.master.title("Video GUI")
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)
        self.master.resizable(False, False)
        self.master.bind("<Escape>", lambda e: self.on_close())

        self.master.config(bg="black")

        self.create_widgets(choices)
        self.update_every = int(1000 / FPS)
        self.master.after(self.update_every, self.update_frame)

        self.frame_count = 0

        self.last_landmarks_and_choice = None

    def on_close(self):
        print("Closing the application")
        with open(dataset_json_path, "w") as f:
            for line in dataset_json:
                f.write(json.dumps(line) + "\n")
        cap.release()
        self.master.destroy()

    def create_widgets(self, choices: List[str]):
        font = Font(family='Helvetica', size=20, weight='bold')

        self.image_label = tk.Label(self.master, anchor=tk.CENTER)
        self.image_label.pack()

        # Create a dropdown menu with initial selection
        self.frame = tk.Frame(self.master, bg="black")
        self.frame.pack()

        self.menu_variable = tk.StringVar()
        self.menu_variable.set(choices[0])
        self.dropdown_menu = tk.OptionMenu(self.frame, self.menu_variable, *choices)
        self.dropdown_menu.config(font=font)
        self.dropdown_menu.pack(side=tk.LEFT)
        # Bind dropdown menu change event
        self.dropdown_menu.bind('<<MenuSelect>>', self.on_dropdown_select)

        self.capture_button = tk.Button(self.frame,
                                        text="Capture Coordinates",
                                        command=self.capture_button_click,
                                        font=font, )
        self.capture_button.pack(side=tk.RIGHT)

    def capture_button_click(self):
        if self.last_landmarks_and_choice is not None:
            dataset_json.append(self.last_landmarks_and_choice)
            print(f"Added {self.last_landmarks_and_choice} to the dataset")

    def update_frame(self):
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)

            landmarks = get_landmarks(frame, self.frame_count)
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            if landmarks is not None:
                choice = self.menu_variable.get()
                self.last_landmarks_and_choice = {"landmarks": landmarks.reshape(-1).tolist(), "choice": choice}
                frame = draw_landmarks_on_image(frame, landmarks)
            else:
                self.last_landmarks_and_choice = None
            self.frame_count += 1

            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=image)
            self.image_label.config(image=photo)
            photo.image = photo  # Keep a reference to prevent garbage collection

            self.master.after(self.update_every, self.update_frame)

    def on_dropdown_select(self, event):
        selected_choice = self.menu_variable.get()
        print(f"Selected choice: {selected_choice}")


root = tk.Tk()
root.geometry(f"{WIDTH}x{HEIGHT + 100}")

with open('choices.txt') as f:
    choices = f.read().splitlines()
    choices.sort()
video_gui = VideoGUI(root, choices=choices)
root.mainloop()
