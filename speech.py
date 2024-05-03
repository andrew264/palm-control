import multiprocessing as mp
import os
import time

import chime
import speech_recognition as sr
import torch
import whisper

chime.theme('big-sur')


class SpeechThread(mp.Process):
    def __init__(self, signal_queue: mp.Queue, typewriter_queue: mp.Queue):
        super().__init__()
        self.model_name = "small.en"
        self.whisper_model = None
        self.recognizer = sr.Recognizer()
        self.recognizer.whisper_model = {self.model_name: self.whisper_model, }
        self.signal_queue = signal_queue
        self.typewriter_queue = typewriter_queue

    def run(self):
        print(f"{self.__class__.__name__}'s PID: {os.getpid()}")
        if not self.whisper_model:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"Using {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                print("Using CPU ---------------- WARNING: This will be slow! -----------------")
            start = time.time()
            self.whisper_model = whisper.load_model(name=self.model_name, device=device, ).eval()
            print(f"Loaded whisper model in {time.time() - start:.2f} seconds")
            start = time.time()
            torch.compile(model=self.whisper_model.forward, dynamic=True, fullgraph=True, mode='max-autotune')
            print(f"Compiled in whisper model {time.time() - start:.2f} seconds")
        try:
            while True:
                if not self.signal_queue.empty() and self.signal_queue.get():  # If the signal queue is not empty
                    self.speech_to_text()
                    while not self.signal_queue.empty():  # Clear the queue
                        self.signal_queue.get_nowait()
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass

    def speech_to_text(self, ):
        print("Listening...")
        with sr.Microphone() as source:
            try:
                self.recognizer.adjust_for_ambient_noise(source)
                start = time.time()
                chime.success()
                audio = self.recognizer.listen(source)
                print(f"Listened for {time.time() - start:.2f} seconds")
                start = time.time()
                text: str = self.recognizer.recognize_whisper(audio, model=self.model_name, language="en").strip()
                chime.success()
                print(f"Recognized: {text} in {time.time() - start:.2f} seconds")
                self.typewriter_queue.put_nowait(text, )
            except sr.UnknownValueError as e:
                print(e)
                chime.error()
            except sr.RequestError as e:
                print("Speech recognition service unavailable\n", e)
                chime.error()
            except KeyboardInterrupt:
                pass
        print("Done listening")
