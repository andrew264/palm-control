from typing import Union

import cv2
import numpy as np
import torch
from onnxruntime import InferenceSession

from typin import HAND_CONNECTIONS


def _landmarks_list_to_array(landmark_list) -> np.ndarray:
    return np.array([[point.x, point.y, point.z] for point in landmark_list])


def _rescale_landmarks(landmarks: Union[np.ndarray, torch.Tensor], height: int, width: int) -> list:
    return [[int(point[0] * width), int(point[1] * height)] for point in landmarks]


def draw_landmarks_on_image(_img: np.ndarray, _points: Union[np.ndarray, torch.Tensor]):
    _img = np.copy(_img)
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


def load_onnx_model(model_path: str) -> InferenceSession:
    import onnxruntime as ort
    ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    return ort_session


def run_inference_on_onnx_model(model: InferenceSession, input_data: np.ndarray) -> np.ndarray:
    input_data = np.expand_dims(normalize_landmarks(input_data).astype(np.float32), axis=0)
    input_data = {k.name: v for k, v in zip(model.get_inputs(), [input_data])}
    return model.run(None, input_data)[0]


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks in 3D space.
    """
    assert landmarks.shape[1] == 3, "Landmarks must be in 3D space"

    min_vals = np.min(landmarks, axis=0)
    max_vals = np.max(landmarks, axis=0)

    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0

    return (landmarks - min_vals) / ranges


def batch_normalize_landmarks(landmarks: torch.Tensor) -> torch.Tensor:
    """
    Apply normalization to landmarks in 3D space.
    Must be in Batch x 21 x 3 format.
    """
    min_vals = torch.min(landmarks, dim=-2)[0]
    max_vals = torch.max(landmarks, dim=-2)[0]

    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0

    return (landmarks - min_vals.unsqueeze(-2)) / ranges.unsqueeze(-2)


def batch_rotate_points(points: torch.Tensor, max_angle: int) -> torch.Tensor:
    """
    Apply random rotate to points around x, y, and z axes.
    Must be in Batch x 21 x 3 format.
    """
    batch_size = points.shape[0]

    angle_x = torch.deg2rad(torch.randint(-max_angle, max_angle, (batch_size, 1)))
    angle_y = torch.deg2rad(torch.randint(-max_angle, max_angle, (batch_size, 1)))
    angle_z = torch.deg2rad(torch.randint(-max_angle // 2, max_angle // 2, (batch_size, 1)))

    cos_x, sin_x = torch.cos(angle_x), torch.sin(angle_x)
    cos_y, sin_y = torch.cos(angle_y), torch.sin(angle_y)
    cos_z, sin_z = torch.cos(angle_z), torch.sin(angle_z)
    ones = torch.ones((batch_size, 1))
    zeros = torch.zeros((batch_size, 1))

    Rx = torch.cat([ones, zeros, zeros,
                    zeros, cos_x, -sin_x,
                    zeros, sin_x, cos_x], dim=1).reshape(batch_size, 3, 3)

    Ry = torch.cat([cos_y, zeros, sin_y,
                    zeros, ones, zeros,
                    -sin_y, zeros, cos_y], dim=1).reshape(batch_size, 3, 3)

    Rz = torch.cat([cos_z, -sin_z, zeros,
                    sin_z, cos_z, zeros,
                    zeros, zeros, ones], dim=1).reshape(batch_size, 3, 3)

    return torch.matmul(torch.matmul(torch.matmul(points, Rz), Ry), Rx)


def get_gesture_class_labels(file_path: str) -> list[str]:
    with open(file_path, "r") as file:
        labels = file.read().splitlines()
        labels = [label.strip() for label in labels]
        return sorted(labels)
