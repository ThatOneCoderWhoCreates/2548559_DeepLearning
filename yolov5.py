import torch
import os

if __name__ == "__main__":
    os.system(
        "python train.py --img 640 --batch 8 --epochs 50 "
        "--data ../signature.yaml "
        "--weights yolov5s.pt "
        "--device 0 "
        "--name yolov5_signature"
    )