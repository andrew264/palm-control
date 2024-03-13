import numpy as np

from typin import HandLandmark

# Camera
WIDTH, HEIGHT = 1280, 720
FPS = 30
CAMERA_ID = 0
EMPTY_FRAME = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

# Hand tracking
NUM_HANDS = 1
DEFAULT_TRACKING_SMOOTHNESS: float = 5e-1
DEFAULT_MOUSE_SMOOTHNESS: float = 0.7

# GUI Settings
DEFAULT_SHOW_WEBCAM = False
DEFAULT_POINTER_SOURCE = HandLandmark.WRIST