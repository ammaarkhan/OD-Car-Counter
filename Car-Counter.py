from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture("Videos/cars.mp4") # for video
mask = cv2.imread("mask.png")

model = YOLO("../Yolo-Weights/yolov8n.pt")

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

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask) # overlay canva mask onto video
    results = model(imgRegion, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # finding the bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)


            # finding the confidence value
            conf = math.ceil((box.conf[0]*100))/100

            # class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" and conf > 0.4:
                cv2.rectangle(imgRegion, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cvzone.putTextRect(imgRegion, f'{currentClass} {conf}', (max(0, x1), max(35, y1 - 20)),
                                   scale=1, thickness=1, offset=5)

    # cv2.imshow('Image', img)
    cv2.imshow('ImageRegion', imgRegion)
    cv2.waitKey(0)