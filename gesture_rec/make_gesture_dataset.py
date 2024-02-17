import json
import os
import sys
import tkinter as tk
from tkinter import ttk
from tkinter.font import Font
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image, ImageTk

sys.path.insert(0, '../')

from utils import load_mediapipe_model, draw_landmarks_on_image, normalize_landmarks, get_gesture_class_labels

cap = cv2.VideoCapture(0)
WIDTH, HEIGHT = 1280, 720
FPS = 30
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

detector = load_mediapipe_model(num_hands=1, model_path='../models/hand_landmarker.task')
dataset_json = []
dataset_json_path = "dataset.jsonl"
if os.path.exists(dataset_json_path):
    with open(dataset_json_path, "r") as f:
        dataset_json = [json.loads(line) for line in f]


def get_landmarks(frame: np.ndarray, frame_count: int) -> Optional[np.ndarray]:
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    results = detector.detect_for_video(mp_frame, frame_count)
    hand_landmarks = results.hand_landmarks
    if hand_landmarks:
        np_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks[0]])
        return np_landmarks
    return None


class VideoGUI:
    def __init__(self, choices: List[str], gesture_model=None):
        self.gesture_model = gesture_model
        self.choices = choices
        self.master = tk.Tk()
        self.master.geometry(f"{WIDTH}x{HEIGHT + 100}")
        self.master.title("Create Gesture Dataset | Inference Model")
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)
        self.master.resizable(False, False)
        self.master.bind("<Escape>", lambda e: self.on_close())

        self.master.bind("<space>", lambda e: self.capture_button_click())
        self.master.bind("<BackSpace>", lambda e: self.undo_button_click())
        self.master.bind("<Control-z>", lambda e: self.undo_button_click())

        self.master.config(bg="black")

        self.image_label = None
        self.frame = None
        self.dropdown_menu = None
        self.menu_variable = None
        self.capture_button = None

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
        font = Font(family='Roboto Mono', size=14)

        self.image_label = ttk.Label(self.master)
        self.image_label.pack()

        # Create a dropdown menu with initial selection
        self.frame = ttk.Frame(self.master)
        self.frame.pack(fill="x", pady=10)

        self.menu_variable = tk.StringVar()
        self.menu_variable.set(choices[0])

        self.dropdown_menu = ttk.Combobox(self.frame, state="readonly", textvariable=self.menu_variable,
                                          values=choices, font=font)
        self.dropdown_menu.pack(side="left", padx=(0, 10))
        self.dropdown_menu.bind("<<ComboboxSelected>>", self.on_dropdown_select)

        style = ttk.Style()
        style.configure("TButton", font=font)

        self.capture_button = ttk.Button(self.frame, text="Capture Coordinates", command=self.capture_button_click)
        self.capture_button.pack(side="right")

    @staticmethod
    def undo_button_click():
        if len(dataset_json) > 0:
            dataset_json.pop()
            print(f"Removed the last item from the dataset")

    def capture_button_click(self):
        if self.last_landmarks_and_choice is not None:
            dataset_json.append(self.last_landmarks_and_choice)
            print(f"Added {self.last_landmarks_and_choice} to the dataset")

    def get_top_guesses(self, landmarks: np.ndarray, k: int = 3) -> list[str]:
        assert self.gesture_model is not None, "Gesture model is not loaded"
        landmarks = torch.tensor(normalize_landmarks(landmarks)).float()
        outputs = self.gesture_model(landmarks.unsqueeze(0))
        top_k = torch.topk(outputs, k)
        top_k_labels = [self.choices[i] for i in top_k.indices[0].tolist()]
        top_k_probs = top_k.values[0].tolist()
        return [f"{label} ({prob * 100:.2f}%)" for label, prob in zip(top_k_labels, top_k_probs)]

    def update_frame(self):
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)

            landmarks = get_landmarks(frame, self.frame_count)
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            if landmarks is not None:
                choice = self.menu_variable.get()
                self.last_landmarks_and_choice = {"landmarks": landmarks.reshape(-1).tolist(), "label": choice}
                frame = draw_landmarks_on_image(frame, landmarks)
                if self.gesture_model is not None:
                    guesses = self.get_top_guesses(landmarks)
                    for i, guess in enumerate(guesses):
                        cv2.putText(frame, guess, (10, 50 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
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


if __name__ == '__main__':
    choices_file = "choices.txt"
    if not os.path.exists(choices_file):
        raise FileNotFoundError(f"File {choices_file} not found")
    labels = get_gesture_class_labels(choices_file)

    model_save_path = "../models/gesture_model.pth"
    if not os.path.exists(model_save_path):
        print(f"File {model_save_path} not found; not loading the gesture model")
        gesture_model = None
    else:
        from utils import load_gesture_model

        gesture_model = load_gesture_model(model_save_path, len(labels))
    video_gui = VideoGUI(choices=labels, gesture_model=gesture_model)
    video_gui.master.mainloop()
