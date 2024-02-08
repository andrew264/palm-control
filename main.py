import threading
import time
from typing import Optional, Any

import cv2
import numpy as np
from mmpose.apis import init_model, inference_topdown
from torch import nn

from hand import Hand
from utils import HAND_CONNECTIONS, _rescale_landmarks
from yolo import YOLO

hand = Hand(enable_smoothing=True)
NUM_HANDS = 1


def load_models() -> tuple[YOLO, nn.Module]:
    pose_config = './models/rtm/rtmpose-m_8xb256-210e_hand5-256x256.py'
    pose_weights = './models/rtm/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256.pth'

    # from https://github.com/noorkhokhar99/yolo_hand_detection
    detector = YOLO("models/yolo/cross-hands-yolov4-tiny.cfg",
                    "models/yolo/cross-hands-yolov4-tiny.weights",
                    ["hand"])

    pose_estimator = init_model(pose_config, pose_weights,
                                device='cuda:0', )
    return detector, pose_estimator


def process_one_image(_img, detector: YOLO, pose_estimator: nn.Module) -> Optional[Any]:
    """Return the keypoints of the hand in the image as a numpy array."""
    width, height, inference_time, results = detector.inference(_img)
    results.sort(key=lambda x: x[2])
    bboxes = [x[3:] for x in results[:NUM_HANDS]]

    pose_results = inference_topdown(pose_estimator, _img, bboxes)
    results = pose_results[0].pred_instances['keypoints'][0]
    normed_results = results[:, :2] / np.array([width, height])
    return normed_results


def check_if_hand_is_fucking_up(landmarks: np.ndarray) -> bool:
    pass


def do_hand_tracking(video_device_id: int = 0, show_raw_image: bool = False):
    detector, estimator = load_models()
    cap = cv2.VideoCapture(video_device_id)
    # set mpeg format
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FPS, 45)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            time.sleep(0.1)
            continue
        image = cv2.flip(image, 1)
        landmarks = process_one_image(image, detector, estimator)
        if landmarks is not None:
            if show_raw_image:
                draw_landmarks_on_image(image, landmarks)
            hand.update(landmarks)
        if show_raw_image:
            cv2.imshow('Hand Tracking', image)
            if cv2.waitKey(1) == 27:
                break
        else:
            hand.update(None)


def draw_landmarks_on_image(_img, _points: Optional[np.ndarray]):
    if _points is None:
        return _img
    _points = _rescale_landmarks(_points, _img.shape)
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
        draw_circle(_points[index], 5 if index < 13 else 8, 1)

    return _img


if __name__ == '__main__':
    # do_hand_tracking(show_raw_image=True)
    print("Starting hand tracking thread")
    threading.Thread(target=do_hand_tracking, daemon=True, name="Tracking-Thread").start()
    print("Waiting for hand tracking to start...")
    while hand.is_missing:
        time.sleep(0.5)
    print("Starting main loop")
    while True:
        t = time.time()
        time.sleep(1 / 120)
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        img = draw_landmarks_on_image(img, hand.coordinates)
        hand.do_action()
        fps = 1 / (time.time() - t)
        cv2.putText(img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Hand Tracking', img)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
