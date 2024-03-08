import numpy as np
import cv2
import time

# download the model as plain text as a PROTOTXT file and the trained model as a CAFFEMODEL file from here: https://github.com/djmv/MobilNet_SSD_opencv

# path to the prototxt file with text description of the network architecture
prototxt = "/home/ben/Downloads/MobileNetSSD_deploy.prototxt"
# path to the .caffemodel file with learned network
caffe_model = "/home/ben/Downloads/MobileNetSSD_deploy.caffemodel"

# read a network model (pre-trained) stored in Caffe framework's format
net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)

# dictionary with the object class id and names on which the model is trained
classNames = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
              5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
              10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
              14: 'motorbike', 15: 'person', 16: 'pottedplant',
              17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

# Initialize video capture on the correct device
cap = cv2.VideoCapture(0)

# Ensure the capture device is opened
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    start_time = time.time()  # Start time before the inference
    
    width = frame.shape[1]
    height = frame.shape[0]

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            x_left_bottom = int(detections[0, 0, i, 3] * width)
            y_left_bottom = int(detections[0, 0, i, 4] * height)
            x_right_top = int(detections[0, 0, i, 5] * width)
            y_right_top = int(detections[0, 0, i, 6] * height)

            cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0), 2)

            if class_id in classNames:
                label = f"{classNames[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x_left_bottom, y_left_bottom - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    end_time = time.time()  # End time after the inference
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    fps = 1.0 / (inference_time / 1000)  # Convert inference time back to seconds for FPS calculation

    # Display inference time in milliseconds and FPS
    cv2.putText(frame, f"Inference time: {inference_time:.3f} ms", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    
    cv2.imshow("frame", frame)

    # Break the loop with the ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the capture and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()
