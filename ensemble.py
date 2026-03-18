import torch
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
import numpy as np
import cv2

# ---------------- LOAD MODELS ----------------

model_v8 = YOLO("C:/Users/tanma/runs/detect/yolov8_signature6/weights/best.pt")

model_v5 = torch.hub.load(
    'C:/Users/tanma/OneDrive/Desktop/Christ University/Trimester 3/DL/Lab/2548559_Lab4/yolov5',
    'custom',
    path='C:/Users/tanma/OneDrive/Desktop/Christ University/Trimester 3/DL/Lab/2548559_Lab4/yolov5/runs/train/yolov5_signature3/weights/best.pt',
    source='local'
)

image_path = "C:/Users/tanma/OneDrive/Desktop/Christ University/Trimester 3/DL/Lab/2548559_Lab4/signature/images/val/Frame_160.jpg"

# ---------------- PREDICTIONS ----------------

results_v8 = model_v8(image_path)[0]
results_v5 = model_v5(image_path)

boxes_list = []
scores_list = []
labels_list = []

h, w = results_v8.orig_shape

# YOLOv8
boxes8 = results_v8.boxes.xyxy.cpu().numpy()
scores8 = results_v8.boxes.conf.cpu().numpy()
labels8 = results_v8.boxes.cls.cpu().numpy()
boxes8 = boxes8 / np.array([w, h, w, h])

boxes_list.append(boxes8.tolist())
scores_list.append(scores8.tolist())
labels_list.append(labels8.tolist())

# YOLOv5
boxes5 = results_v5.xyxy[0][:, :4].cpu().numpy()
scores5 = results_v5.xyxy[0][:, 4].cpu().numpy()
labels5 = results_v5.xyxy[0][:, 5].cpu().numpy()
boxes5 = boxes5 / np.array([w, h, w, h])

boxes_list.append(boxes5.tolist())
scores_list.append(scores5.tolist())
labels_list.append(labels5.tolist())

# ---------------- ENSEMBLE ----------------

boxes, scores, labels = weighted_boxes_fusion(
    boxes_list,
    scores_list,
    labels_list,
    weights=[1, 1],
    iou_thr=0.5
)

# ---------------- DRAW RESULT ----------------

image = cv2.imread(image_path)

for box, score in zip(boxes, scores):
    x1 = int(box[0] * w)
    y1 = int(box[1] * h)
    x2 = int(box[2] * w)
    y2 = int(box[3] * h)

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(image, f"Ensemble {score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2)

cv2.imwrite("ensemble_output.jpg", image)

print("Ensemble image saved as ensemble_output.jpg")