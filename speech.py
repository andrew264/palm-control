import time
from threading import Thread

import chime
import pyautogui
import speech_recognition as sr
from whisper import Whisper

chime.theme('big-sur')


class SpeechThread(Thread):
    def __init__(self, model: Whisper, model_name: str):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.recognizer = sr.Recognizer()
        self.recognizer.whisper_model = {self.model_name: self.model, }
        self.transcribing = False

    def run(self):
        while True:
            if self.transcribing:
                self.speech_to_text()
                self.transcribing = False
            time.sleep(1)

    def speech_to_text(self, ):
        print("Listening...")
        with sr.Microphone() as source:
            try:
                self.recognizer.adjust_for_ambient_noise(source)
                chime.success()
                audio = self.recognizer.listen(source)
                text: str = self.recognizer.recognize_whisper(audio, model=self.model_name).strip()
                chime.success()
                print(f"Recognized: {text}")
                pyautogui.typewrite(text, interval=0.1, _pause=True)
            except sr.UnknownValueError as e:
                print(e)
                chime.error()
            except sr.RequestError as e:
                print("Speech recognition service unavailable\n", e)
                chime.error()
            except RuntimeError as e:
                print("pyautogui killed itself ig\n", e)
                chime.error()
        print("Done listening")
