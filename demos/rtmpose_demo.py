import cv2
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.utils import adapt_mmdet_pipeline

det_cat_id = 0
bbox_thr = 0.2
draw_heatmap = False
draw_bbox = True
show_kpt_idx = False
skeleton_style = 'mmpose'
show = True
kpt_thr = 0.1
nms_thr = 0.3

det_config = '../models/rtm/rtmdet_nano_320-8xb32_hand.py'
det_weights = '../models/rtm/rtmdet_nano_8xb32-300e_hand.pth'

pose_config = '../models/rtm/rtmpose-m_8xb256-210e_hand5-256x256.py'
pose_weights = '../models/rtm/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256.pth'


def process_one_image(img,
                      detector,
                      pose_estimator, ):
    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    score = pred_instance[0]['scores'][0]
    if score < bbox_thr:
        return None
    bboxes = pred_instance[0]['bboxes']

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    return pose_results[0].pred_instances['keypoints'][0]


def main():
    detector = init_detector(det_config, det_weights, device='cuda:0')
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    pose_estimator = init_pose_estimator(pose_config,
                                         pose_weights,
                                         device='cuda:0',
                                         cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FPS, 60)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        frame = cv2.flip(frame, 1)

        landmarks = process_one_image(frame, detector, pose_estimator)

        if landmarks is not None:
            print(landmarks)
        else:
            print('No hand detected')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == '__main__':
    main()
