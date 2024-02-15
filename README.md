# palm-control

This is a simple python script that uses the `mediapipe` library to detect the hand and fingers and control the mouse
pointer using the palm and fingers.

# Setup

-

Download [MODEL LINK](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)
- place the model in the `./models` directory

- create a virtual environment using `python -m venv venv`
- install the required packages using `pip install -r requirements.txt`
- run the script using `python app.py`
- to exit the script press `esc` key

# TODO (what I think I need to do)

- [x] use `mediapipe` to get hand cords (easy)
- [x] use `Kalman filtering` to smooth the outputs (kinda easy)
- [x] map them fingers to mouse cursor movements (the hard part) its okay for now
- [x] map clicks and drags to the fingers (easy)
- [x] speech to text (for typing) (hard) (it kainda works; if pyautogui works)
- [ ] profit ??