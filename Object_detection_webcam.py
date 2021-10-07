######################### READ WEBCAM  ############################
# import cv2
# frameWidth = 640
# frameHeight = 480
# cap = cv2.VideoCapture(0)
# cap.set(3, frameWidth)
# cap.set(4, frameHeight)
# while True:
#     success, img = cap.read()
#     cv2.imshow("Result", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

import cv2
import numpy as np
import yaml

# loading data paths from config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

weight_path = config['yolo']['weight_path']
cfg_path = config['yolo']['cfg_path']
class_path = config['yolo']['class_path']

# loading yolo network
net = cv2.dnn.readNetFromDarknet(cfg_path, weight_path)

# loading classes
classes = []
with open(class_path, 'r') as f:
    classes = f.read().splitlines()

# loading webcam
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0) # webcam
cap.set(3, frameWidth)
cap.set(4, frameHeight)

while True:

    _, frame = cap.read()

    height, width, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)

    # for b in blob:
    #     for n, blob_image in enumerate(b):
    #         cv2.imshow(str(n), blob_image)

    net.setInput(blob)

    # output_layers_name = net.getUnconnectedOutLayersNames()
    # layer_outputs = net.forward(output_layers_name)
    ln = net.getLayerNames()
    ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]
    layer_outputs = net.forward(ln)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:  # detection contains 4 hunting box offset(x,y,w,h), 1 box confidence and 80 classes prediction (85 total)
            scores = detection[5:]  # all 80 class predictions
            class_id = np.argmax(scores)  # find out the location that contain highest scores
            confidence = scores[class_id] # assign the highest score into confidence

            if confidence > 0.5:
                center_x = int(detection[0]*width) # multiple with width cuz x is normalized and we want to get back origin 
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2) # extract upper left corners positions
                y = int(center_y - h/2)


                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

                # use non maximum suppressions(NMS) to only keep their highest scores boxes

    # print(len(boxes)) # check how many boxes are being detected

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) # score_thershold = 0.5 sames as confidence, loan maximun thershold=0.4(default)

    # print(indexes.flatten()) # show how many boxes are redundant

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    # ensure at least one detection exists
    if len(indexes) > 0:
        # loop over the indexes
        for i in indexes.flatten():
            # extract the bounding box coordinates
            # x, y, w, h = boxes[i]
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)

    cv2.imshow('Webcam', frame)   

    # key= cv2.waitKey(0)
    # if  key == 27: # esc key
    #     break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()