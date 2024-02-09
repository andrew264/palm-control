import time

import cv2
import numpy as np
from mmpose.apis import init_model, inference_topdown
from torch import nn

from main import draw_landmarks_on_image
from utils import KalmanFilterObj
from yolo import YOLO

NUM_HANDS = 1
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector_filter = KalmanFilterObj(4, enable_smoothing=True)


def do_hand_detection(detector: YOLO):
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            time.sleep(0.1)
            continue
        image = cv2.flip(image, 1)
        width, height, inference_time, results = detector.inference(image)
        if results:
            results.sort(key=lambda x: x[2])  # sort by confidence
            bboxes = [x[3:] for x in results]  # confidence threshold
            print(f"Confidence: {results[0][2]:.2f} | Inference time: {inference_time:.2f}ms")
            if len(bboxes) == 0:
                continue
            bboxes = bboxes[:NUM_HANDS]
            # increase the bounding box size
            bboxes = [(x - 100, y - 100, w + 200, h + 200) for x, y, w, h in bboxes]
            detector_filter.update(np.array(bboxes[0]))
            bboxes = detector_filter.predict()
            bboxes = bboxes.astype(int).tolist()
            for x, y, w, h in bboxes:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Hand Detection", image)
        time.sleep(1 / 60)
        if cv2.waitKey(1) == 27:  # ESC key
            break


def detection_with_pose(detector: YOLO, pose_estimator: nn.Module):
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            time.sleep(0.1)
            continue
        image = cv2.flip(image, 1)
        o_image = image.copy()
        width, height, inference_time, results = detector.inference(o_image)
        if results:
            results.sort(key=lambda x: x[2])  # sort by confidence
            print(f"Confidence: {results[0][2]:.2f} | Inference time: {inference_time:.2f}ms")
            bboxes = [x[3:] for x in results]
            if len(bboxes) == 0:
                continue
            bboxes = bboxes[:NUM_HANDS]
            bboxes = [(x - 75, y - 75, w + 150, h + 150) for x, y, w, h in bboxes]
            detector_filter.update(np.array(bboxes[0]))
        bboxes = detector_filter.predict()
        if bboxes is None:
            continue
        bboxes = bboxes.astype(int).tolist()
        x, y, w, h = bboxes
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        pose_results = inference_topdown(pose_estimator, o_image, [bboxes], bbox_format='xywh')
        if pose_results:
            results = pose_results[0].pred_instances['keypoints'][0]
            normed_results = results[:, :2] / np.array([width, height])

            draw_landmarks_on_image(image, normed_results)

        cv2.imshow("Hand Detection", image)
        time.sleep(1 / 60)
        if cv2.waitKey(1) == 27:
            break


if __name__ == '__main__':
    # detector
    det = YOLO("models/yolo/cross-hands-yolov4-tiny.cfg",
               "models/yolo/cross-hands-yolov4-tiny.weights",
               ["hand"], confidence=0, threshold=0.9)
    # pose
    pose_config = './models/rtm/rtmpose-m_8xb256-210e_hand5-256x256.py'
    pose_weights = './models/rtm/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256.pth'
    pose = init_model(pose_config, pose_weights, device='cuda:0', )
    # do_hand_detection(det)
    detection_with_pose(det, pose)
    cv2.destroyAllWindows()
