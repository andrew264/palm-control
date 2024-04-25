import os
import time
import tkinter as tk
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from tkinter import ttk

import sv_ttk
from PIL import Image, ImageTk

from constants import DEFAULT_TRACKING_SMOOTHNESS, HEIGHT, WIDTH, DEFAULT_MOUSE_SMOOTHNESS, EMPTY_FRAME
from event_processor import EventProcessor
from typin import HandLandmark, GUIEvents

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Palm Control GUI")
        self.root.geometry(f"{WIDTH}x{HEIGHT + 150}")
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

        # Queues
        self.tracking_image = SharedMemory(create=True, size=EMPTY_FRAME.nbytes)
        self.gui_event_queue = Queue(maxsize=10)
        self.event_processor = EventProcessor(self.gui_event_queue, tracking_image_name=self.tracking_image.name)

        self.create_widgets()
        self.update_frame()

    def on_close(self):
        print("Closing the application")
        self.gui_event_queue.put(GUIEvents.EXIT)
        self.tracking_image.unlink()
        self.tracking_image.close()
        self.root.destroy()
        exit(0)

    def create_widgets(self):
        self.tracking_image_label = ttk.Label(self.root, style="TLabel")
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
                                                    style="TCheckbutton", command=self.update_show_webcam)
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

    def update_tracking_smoothness(self, value):
        self.gui_event_queue.put((GUIEvents.TRACKING_SMOOTHNESS, float(value)))

    def update_mouse_smoothness(self, value):
        self.gui_event_queue.put((GUIEvents.MOUSE_SMOOTHNESS, float(value)))

    def update_mouse_pointer(self, event):
        self.gui_event_queue.put((GUIEvents.MOUSE_POINTER, self.mouse_pointer_source.get()))

    def update_show_webcam(self):
        boolean_value = True if self.show_webcam_var.get() == 1 else False
        self.gui_event_queue.put((GUIEvents.SHOW_WEBCAM, boolean_value))

    def update_frame(self):
        img = Image.frombytes("RGB", (WIDTH, HEIGHT), self.tracking_image.buf)
        img = ImageTk.PhotoImage(image=img)
        self.tracking_image_label.config(image=img)
        self.tracking_image_label.image = img
        self.root.after(10, self.update_frame)

    def run(self):
        start = time.time()
        self.event_processor.start()
        print(f"Event Processor started in {time.time() - start:.2f} seconds")
        self.root.mainloop()


if __name__ == '__main__':
    app = GUI()
    app.run()
