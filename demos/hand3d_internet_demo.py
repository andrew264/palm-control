import time

import cv2
import mmcv
import numpy as np
from mmengine import VISUALIZERS
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import PoseDataSample, merge_data_samples

config_file = '../models/internet_res50_4xb16-20e_interhand3d-256x256.py'
pose3d_weights = '../models/res50_intehand3dv1.0_all_256x256.pth'
kpt_thr = 0.1
show_kpt_idx = False
show = True
disable_rebase_keypoint = False


def process_one_image(img, model, visualizer=None, show_interval: float = 0.01):
    """Visualize predicted keypoints of one image."""
    # inference a single image
    pose_results = inference_topdown(model, img)
    # post-processing
    pose_results_2d = []
    for idx, res in enumerate(pose_results):
        pred_instances = res.pred_instances
        keypoints = pred_instances.keypoints
        rel_root_depth = pred_instances.rel_root_depth
        scores = pred_instances.keypoint_scores
        hand_type = pred_instances.hand_type

        res_2d = PoseDataSample()
        gt_instances = res.gt_instances.clone()
        pred_instances = pred_instances.clone()
        res_2d.gt_instances = gt_instances
        res_2d.pred_instances = pred_instances

        # add relative root depth to left hand joints
        keypoints[:, 21:, 2] += rel_root_depth

        # set joint scores according to hand type
        scores[:, :21] *= hand_type[:, [0]]
        scores[:, 21:] *= hand_type[:, [1]]
        # normalize kpt score
        if scores.max() > 1:
            scores /= 255

        res_2d.pred_instances.set_field(keypoints[..., :2].copy(), 'keypoints')

        # rotate the keypoint to make z-axis correspondent to height
        # for better visualization
        vis_R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        keypoints[..., :3] = keypoints[..., :3] @ vis_R

        # rebase height (z-axis)
        if not disable_rebase_keypoint:
            valid = scores > 0
            keypoints[..., 2] -= np.min(
                keypoints[valid, 2], axis=-1, keepdims=True)

        pose_results[idx].pred_instances.keypoints = keypoints
        pose_results[idx].pred_instances.keypoint_scores = scores
        pose_results_2d.append(res_2d)

    data_samples = merge_data_samples(pose_results)
    data_samples_2d = merge_data_samples(pose_results_2d)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            det_data_sample=data_samples_2d,
            draw_gt=False,
            draw_bbox=True,
            kpt_thr=kpt_thr,
            convert_keypoint=False,
            axis_azimuth=-115,
            axis_limit=200,
            axis_elev=15,
            show_kpt_idx=show_kpt_idx,
            show=show,
            wait_time=show_interval)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)


def main():
    model = init_model(config_file, pose3d_weights, device='cuda:0')
    # init visualizer
    model.cfg.visualizer.radius = 4
    model.cfg.visualizer.line_width = 2

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(model.dataset_meta)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FPS, 60)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        frame = cv2.flip(frame, 1)
        # frame = cv2.resize(frame, (256, 256))
        pred_instances = process_one_image(frame, model, visualizer,
                                           0.001)
        if pred_instances is not None:
            print(pred_instances.keypoints)
        if cv2.waitKey(5) & 0xFF == 27:
            break
        time.sleep(1 / 30)
    cap.release()


if __name__ == '__main__':
    main()
