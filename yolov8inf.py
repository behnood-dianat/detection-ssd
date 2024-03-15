import cv2
import numpy as np
from ultralytics import YOLO
import torch

# Check if CUDA is available, and set the device to GPU (cuda:0) or CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the YOLOv8 model (the model will automatically use CUDA if it's available)
model = YOLO('yolov8n.pt')  # Assuming yolov8n.pt is in the current directory

# Open your webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

if not cap.isOpened():
    print("Error opening webcam")
    exit()

while True:
    success, img = cap.read()
    if not success:
        print("Could not read frame from webcam. Exiting...")
        break

    # Preprocess the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 480))  # Resize to be compatible with the model
    img = np.transpose(img, (2, 0, 1))  # Convert HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension BCHW
    img = img / 255.0  # Normalize to [0, 1]
    img = torch.from_numpy(img).float()
    img = img.to(device)  # Move the tensor to the GPU

    # Run YOLOv8 inference on the frame
    results = model(img)
    print(f"Input tensor is on: {img.device}")

    # Assuming 'results[0].plot()' returns a NumPy array in RGB format
    annotated_img = results[0].plot()

    # Convert RGB to BGR for OpenCV display
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)

    # Display the resulting image
    cv2.imshow('YOLOv8 Object Detection', annotated_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
