from typing import TypeVar

import cv2
import numpy as np
import torch

from typin import HAND_CONNECTIONS


def _landmarks_list_to_array(landmark_list) -> np.ndarray:
    return np.array([[point.x, point.y, point.z] for point in landmark_list])


def _rescale_landmarks(landmarks: np.ndarray, height: int, width: int) -> list:
    return [[int(point[0] * width), int(point[1] * height)] for point in landmarks]


def draw_landmarks_on_image(_img: np.ndarray, _points: np.ndarray):
    _points = _rescale_landmarks(_points, *_img.shape[:2])
    colors = [(0, 0, 0), (255, 255, 255)]

    def draw_line(_start, _end, thickness):
        cv2.line(_img, tuple(_start), tuple(_end), colors[0], thickness + 4)
        cv2.line(_img, tuple(_start), tuple(_end), colors[1], thickness)

    def draw_circle(center, radius, thickness):
        cv2.circle(_img, tuple(center), radius + 3, colors[0], -1)
        cv2.circle(_img, tuple(center), radius, colors[1], thickness)

    for start, end in HAND_CONNECTIONS:
        draw_line(_points[start], _points[end], 6)

    for index in range(len(_points)):
        draw_circle(_points[index], 5, 1)

    return _img


def draw_mp_landmarks_on_image(rgb_image, detection_result):
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2

    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image


def load_mediapipe_model(num_hands: int = 2, model_path: str = './models/hand_landmarker.task'):
    import mediapipe as mp
    from mediapipe.tasks import python
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionRunningMode.VIDEO,
        num_hands=num_hands,
        min_tracking_confidence=0.3)
    detector = HandLandmarker.create_from_options(options)
    return detector


def load_gesture_model(model_path: str, num_classes: int) -> torch.nn.Module:
    from gesture_network import GestureFFN
    model = GestureFFN(input_size=21 * 3, hidden_size=256, output_size=num_classes)
    try:
        model.load_state_dict(torch.load(model_path))
    except RuntimeError:
        print(f"Failed to load weights possibly due to a mismatch in the number of classes")
        print(f"Loading model without weights")
    model.cpu()
    model.eval()
    return model


T = TypeVar('T', list, np.ndarray, torch.Tensor)


def normalize_landmarks(landmarks: T) -> T:
    if isinstance(landmarks, np.ndarray):
        n_landmarks = landmarks.copy()
        n_axes = n_landmarks.shape[1]
        for axis in range(n_axes):
            min_value = np.min(n_landmarks[:, axis])
            max_value = np.max(n_landmarks[:, axis])
            n_landmarks[:, axis] = (n_landmarks[:, axis] - min_value) / (max_value - min_value)
        return n_landmarks
    elif isinstance(landmarks, torch.Tensor):
        n_landmarks = landmarks.float().clone()
        n_axes = n_landmarks.shape[1]
        for axis in range(n_axes):
            min_value = torch.min(n_landmarks[:, axis])
            max_value = torch.max(n_landmarks[:, axis])
            n_landmarks[:, axis] = (n_landmarks[:, axis] - min_value) / (max_value - min_value)
        return n_landmarks
    else:
        raise ValueError(f"Unsupported type {type(landmarks)}")


def get_gesture_class_labels(file_path: str) -> list[str]:
    with open(file_path, "r") as file:
        labels = file.read().splitlines()
        labels = [label.strip() for label in labels]
        return sorted(labels)
