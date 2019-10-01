"""
This python file operates OpenCV(3.4.2+)'s Darknet framework in the DNN module to run YOLOv3 object detection.
"""
import numpy as np
import cv2
import os
import time

yolo_dir = '/home/bowenl2016/yolov3'    # yolo file directory
weightsPath = os.path.join(yolo_dir, 'yolov3.weights')    # yolo weights
configPath = os.path.join(yolo_dir, 'yolov3.cfg')    # configuration file
labelsPath = os.path.join(yolo_dir, 'coco.names')    # labels file
imgPath = os.path.join(yolo_dir, 'test.jpg')    # test 4K image
CONFIDENCE = 0.5    # minimum probability to filter weak detections
THRESHOLD = 0.4    # non-maximum suppression threshold

# load net, confign and weights
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
print("YOLO loaded from disk.")

# load the image, convert it to blob format, and send into net input layer
img = cv2.imread(imgPath)
blobImg = cv2.dnn.blobFromImage(img, 1.0/255.0, (416, 416), None, True, False)
net.setInput(blobImg)

# get net output layers info and set forward
# record time span
outInfo = net.getUnconnectedOutLayersNames()
start = time.time()
layerOutputs = net.forward(outInfo)    # get each output layer's bounding boxes info
end = time.time()
print(" YOLO took {:.6f} seconds".format(end - start))

# get image dimension
(H, W) = img.shape[:2]

# initiate output arrays
boxes = []
confidences = []
classIDs = []

# filter low confidence score results
for out in layerOutputs:    # for each output layer
    for detection in out:    # for each bounding box
        scores = detection[5:]    # scores of each class
        classID = np.argmax(scores)    # highest score id is classID
        confidence = scores[classID]    # get confidence score

	# filtering
        if confidence > CONFIDENCE:
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# NMS filter
idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

# acquire labels list
with open(labelsPath, 'rt') as f:
    labels = f.read().rstrip('\n').split('\n')
# make demo image show
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
if len(idxs) > 0:
    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
        cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.namedWindow('YOLO Test', cv2.WINDOW_NORMAL)
cv2.imshow('YOLO Test', img)
cv2.waitKey(0)
