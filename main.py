import threading
import time
from typing import Optional, Any

import cv2
import numpy as np
from mmpose.apis import init_model, inference_topdown
from torch import nn

from hand import Hand
from utils import KalmanFilterObj, draw_landmarks_on_image
from yolo import YOLO

hand = Hand(enable_smoothing=True)
NUM_HANDS = 1
det_filter = KalmanFilterObj(4, enable_smoothing=True)

cap = cv2.VideoCapture(0)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)


def load_models() -> tuple[YOLO, nn.Module]:
    pose_config = './models/rtm/rtmpose-m_8xb256-210e_hand5-256x256.py'
    pose_weights = './models/rtm/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256.pth'

    # from https://github.com/noorkhokhar99/yolo_hand_detection
    detector = YOLO("models/yolo/cross-hands-yolov4-tiny.cfg",
                    "models/yolo/cross-hands-yolov4-tiny.weights",
                    ["hand"], confidence=0.1, threshold=0.2)

    pose_estimator = init_model(pose_config, pose_weights,
                                device='cuda:0', )
    return detector, pose_estimator


def get_bbox_from_det(detector: YOLO, _img) -> Optional[Any]:
    """
    Return the bounding box of the hand in the image as a numpy array.
    Format: [x, y, w, h]
    """
    width, height, inference_time, results = detector.inference(_img)
    if results:
        results.sort(key=lambda x: x[2])  # sort by confidence
        bboxes = results[:NUM_HANDS]
        bbox = bboxes[0][3:]
        bbox = (bbox[0] - 75, bbox[1] - 75, bbox[2] + 150, bbox[3] + 150)
        det_filter.update(np.array(bbox))
    return det_filter.predict()


def process_one_image(_img, detector: YOLO, pose_estimator: nn.Module) -> Optional[Any]:
    """Return the keypoints of the hand in the image as a numpy array."""
    bbox = get_bbox_from_det(detector, _img)
    if bbox is not None:
        width, height = _img.shape[:2]
        pose_results = inference_topdown(pose_estimator, _img, [bbox], bbox_format='xywh')
        if pose_results:
            results = pose_results[0].pred_instances['keypoints'][0]
            if results.min() < 1 or results.max() > CAMERA_WIDTH + 200:
                return None
            normed_results = results[:, :2] / np.array([width, height])
            return normed_results if is_hand_good(normed_results) else None
    return None


def is_hand_good(landmarks: np.ndarray, threshold: int = 5) -> bool:
    """
    landmarks shape [21, 2] float32
    if more the threshold points are along the same x-axis return false
    if more the threshold points are along the same y-axis return false
    """
    x, y = landmarks[:, 0], landmarks[:, 1]
    x = np.round(x, 2)
    y = np.round(y, 2)
    x_unique, x_counts = np.unique(x, return_counts=True)
    y_unique, y_counts = np.unique(y, return_counts=True)
    if x_counts.max() > threshold or y_counts.max() > threshold:
        return False
    return True


def do_hand_tracking():
    detector, estimator = load_models()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            time.sleep(0.1)
            continue
        image = cv2.flip(image, 1)
        landmarks = process_one_image(image, detector, estimator)
        hand.update(landmarks)


if __name__ == '__main__':
    # do_hand_tracking()
    print("Starting hand tracking thread")
    threading.Thread(target=do_hand_tracking, daemon=True, name="Tracking-Thread").start()
    print("Waiting for hand tracking to start...")
    while hand.is_missing:
        time.sleep(0.5)
    print("Starting main loop")
    while True:
        t = time.time()
        time.sleep(1 / 120)
        img = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        if not hand.is_missing:
            img = draw_landmarks_on_image(img, hand.coordinates)
            hand.do_action(disable_clicks=True)
        fps = 1 / (time.time() - t)
        cv2.putText(img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Hand Tracking', img)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
