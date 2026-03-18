from ultralytics import YOLO
import torch

def main():
    print("GPU Available:", torch.cuda.is_available())
    print("Using Device:", torch.cuda.get_device_name(0))

    model = YOLO("yolov8s.pt")

    model.train(
        data="signature.yaml",
        epochs=50,
        imgsz=640,
        batch=8,    
        device=0,
        name="yolov8_signature"
    )

if __name__ == "__main__":
    main()