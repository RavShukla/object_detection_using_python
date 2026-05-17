#my first model very basic and less accurate 
#made by Gaurav Shukla 

import cv2
import matplotlib.pyplot as plt
import time

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'







model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
file_name = 'labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

image = cv2.imread('sample_input.jpg')
if image is None:
    raise IOError("Image not found or unable to load.")

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ClassIndex, confidence, bbox = model.detect(image, confThreshold=0.5)
print(ClassIndex)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
if len(ClassIndex) != 0:
    for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
        if 0 < ClassInd <= len(classLabels):
            cv2.rectangle(image, boxes, (255, 0, 0), 2)
            cv2.putText(image, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('Cant open video')

prev_time = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time

    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.5)
    print(ClassIndex)

    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if 0 < ClassInd <= len(classLabels):
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(frame, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=1.5, color=(0, 255, 0), thickness=2)

    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 50), font, 1.5, (0, 255, 255), 2)

    cv2.imshow('Obj detection', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
