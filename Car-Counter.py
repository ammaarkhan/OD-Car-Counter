import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

cap = cv2.VideoCapture("Videos/cars.mp4")  # for video
mask = cv2.imread("mask.png")

model = YOLO("../Yolo-Weights/yolov8n.pt")

# tracking:
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

lineL = [340, 337, 473, 337]
lineM = [480, 337, 590, 337]
lineR = [597, 337, 700, 337]

numCountL = []
numCountM = []
numCountR = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)  # overlay canva mask onto video
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # finding the bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # finding the confidence value
            conf = math.ceil((box.conf[0] * 100)) / 100

            # class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" and conf > 0.4:
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1 - 20)),
                #                    scale=1, thickness=1, offset=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultTracker = tracker.update(detections)
    cv2.line(img, (lineL[0], lineL[1]), (lineL[2], lineL[3]), (0, 0, 255), 5)
    cv2.line(img, (lineM[0], lineM[1]), (lineM[2], lineM[3]), (0, 0, 255), 5)
    cv2.line(img, (lineR[0], lineR[1]), (lineR[2], lineR[3]), (0, 0, 255), 5)

    for result in resultTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(0, 255, 0))
        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=6)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if lineL[0] < cx < lineL[2] and lineL[1] - 20 < cy < lineL[1] + 20:
            if numCountL.count(id) == 0:
                numCountL.append(id)
                cv2.line(img, (lineL[0], lineL[1]), (lineL[2], lineL[3]), (0, 255, 0), 5)
        elif lineM[0] < cx < lineM[2] and lineM[1] - 20 < cy < lineM[1] + 20:
            if numCountM.count(id) == 0:
                numCountM.append(id)
                cv2.line(img, (lineM[0], lineM[1]), (lineM[2], lineM[3]), (0, 255, 0), 5)
        elif lineR[0] < cx < lineR[2] and lineR[1] - 20 < cy < lineR[1] + 20:
            if numCountR.count(id) == 0:
                numCountR.append(id)
                cv2.line(img, (lineR[0], lineR[1]), (lineR[2], lineR[3]), (0, 255, 0), 5)

    cvzone.putTextRect(img, f'Left Lane: {len(numCountL)}', (20, 40))
    cvzone.putTextRect(img, f'Mid Lane: {len(numCountM)}', (20, 80))
    cvzone.putTextRect(img, f'Right Lane: {len(numCountR)}', (20, 120))

    cv2.imshow('Image', img)
    # cv2.imshow('ImageRegion', imgRegion)
    cv2.waitKey(1)
