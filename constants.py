import numpy as np

from typin import HandLandmark

# Camera
WIDTH, HEIGHT = 1920, 1080
FPS = 60
CAMERA_ID = 0
EMPTY_FRAME = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

# Hand tracking
NUM_HANDS = 1
DEFAULT_TRACKING_SMOOTHNESS: float = 4e-1
DEFAULT_MOUSE_SMOOTHNESS: float = 0.75

# GUI Settings
DEFAULT_SHOW_WEBCAM = False
DEFAULT_POINTER_SOURCE = HandLandmark.INDEX_FINGER_TIP
