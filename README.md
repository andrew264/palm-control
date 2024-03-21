# palm-control

- This is a python application that uses the `mediapipe` library to detect the hand and fingers and control the mouse
  pointer using the palm and fingers.
- It uses a Neural Network to classify the gestures and perform the corresponding action.
- Uses `ONNX` runtime to run the model.
- Uses `Tkinter` for the GUI.
- Uses `multi-threading` to run the model and the GUI separately.
- Has audio recognition using `OpenAI's Whisper` model.
- Runs on `Windows` and `Linux`.

# Setup

Download [MODEL LINK](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

- place the model in the `./models` directory

- create a virtual environment using `python -m venv venv`
- install the required packages using `pip install -r requirements.txt`
- run the script using `python app.py`
- to exit the script press `esc` key

# Things done

- [x] Hand detection
- [x] Filtering the hand cords
- [x] Speech to text using Whisper
- [x] GUI for the app
- [x] Mouse movement
- [x] Gesture classification
- [x] Training code and Dataset creation code
- [x] ONNX model inference

# TODO (what I think I need to do)

- [ ] Add more gestures ???
- [ ] profit ???