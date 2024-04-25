import multiprocessing as mp
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
        if not self.whisper_model:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"Using {torch.cuda.get_device_name()}")
                dtype = torch.float16
            else:
                device = torch.device("cpu")
                dtype = torch.float32
                print("Using CPU ---------------- WARNING: This will be slow! -----------------")
            start = time.time()
            self.whisper_model = whisper.load_model(name=self.model_name, device=device, ).eval()
            self.whisper_model = self.whisper_model.to(dtype=dtype)
            print(f"Loaded whisper model in {time.time() - start:.2f} seconds")
            start = time.time()
            torch.compile(model=self.whisper_model.forward, fullgraph=True, mode='max-autotune')
            print(f"Compiled in whisper model {time.time() - start:.2f} seconds")
        while True:
            if not self.signal_queue.empty() and self.signal_queue.get():  # If the signal queue is not empty
                self.speech_to_text()
                while not self.signal_queue.empty():  # Clear the queue
                    self.signal_queue.get()
            time.sleep(0.5)

    def speech_to_text(self, ):
        print("Listening...")
        with sr.Microphone() as source:
            try:
                self.recognizer.adjust_for_ambient_noise(source)
                chime.success()
                start = time.time()
                audio = self.recognizer.listen(source)
                print(f"Listened for {time.time() - start:.2f} seconds")
                start = time.time()
                text: str = self.recognizer.recognize_whisper(audio, model=self.model_name).strip()
                chime.success()
                print(f"Recognized: {text} in {time.time() - start:.2f} seconds")
                # pyautogui.typewrite(text, interval=0.1, _pause=True) # pyautogui is not thread safe
                self.typewriter_queue.put(text)
            except sr.UnknownValueError as e:
                print(e)
                chime.error()
            except sr.RequestError as e:
                print("Speech recognition service unavailable\n", e)
                chime.error()
            except RuntimeError as e:
                print("pyautogui killed itself ig\n", e)
                chime.error()
            except KeyboardInterrupt:
                pass
        print("Done listening")
