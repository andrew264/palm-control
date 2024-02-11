from threading import Thread

import chime
import pyautogui
import speech_recognition as sr
from whisper import Whisper

chime.theme('big-sur')


def speech_to_text(model: Whisper, model_name: str):
    recognizer = sr.Recognizer()
    recognizer.whisper_model = {
        model_name: model,
    }
    with sr.Microphone() as source:
        try:
            recognizer.adjust_for_ambient_noise(source)
            chime.success()
            audio = recognizer.listen(source)
            text: str = recognizer.recognize_whisper(audio, model=model_name).strip()
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


class SpeechThread(Thread):
    def __init__(self, model: Whisper, model_name: str):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.finished = False
        self.is_running = False

    def run(self):
        self.is_running = True
        speech_to_text(self.model, self.model_name)
        self.finished = True
        self.is_running = False
