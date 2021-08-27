import cv2
import numpy as np

# loading yolo network
net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')

# loading classes
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

# cap = cv2.VideoCapture('test.mp4') # video capture

cap = cv2.VideoCapture(0) # webcam


# img = cv2.imread('test_image.jpeg') # loading test image

while True:

    _, img = cap.read()

    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)

    # for b in blob:
    #     for n, blob_image in enumerate(b):
    #         cv2.imshow(str(n), blob_image)

    net.setInput(blob)

    output_layers_name = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_name)

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

    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]

        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)

    cv2.imshow('Image', img)    
    # cv2.waitKey(0)
    key= cv2.waitKey(0)
    if  key == 27: # esc key
        break

cap.release()
cv2.destroyAllWindows()