######################### READ VIDEO #############################
# import cv2
# frameWidth = 640
# frameHeight = 480
# cap = cv2.VideoCapture("Resources/testVideo.mp4")
# while True:
#     success, img = cap.read()
#     img = cv2.resize(img, (frameWidth, frameHeight))
#     cv2.imshow("Result", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

import cv2
import numpy as np
import imutils
import time
import yaml

# loading data paths from config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

video_path = config.get('video_path')
yolo = config.get('yolo')
weight_path = yolo['weight_path']
cfg_path = yolo['cfg_path']
class_path = yolo['class_path']
output_path = config.get('output_path')

print(video_path)
print(weight_path)
print(cfg_path)
print(class_path)

# loading yolo network
net = cv2.dnn.readNetFromDarknet(cfg_path, weight_path)

# loading classes
classes = []
with open(class_path, 'r') as f:
    classes = f.read().splitlines()

# loading video
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(video_path)
writer = None

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(cap.get(prop))
	print("[INFO] {} total frames in video".format(total))
# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

while True:

    _, frame = cap.read()
    frame = cv2.resize(frame, (frameWidth, frameHeight))
    height, width, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)

    # for b in blob:
    #     for n, blob_image in enumerate(b):
    #         cv2.imshow(str(n), blob_image)

    net.setInput(blob)

    output_layers_name = net.getUnconnectedOutLayersNames()
    start = time.time()
    layer_outputs = net.forward(output_layers_name)
    end = time.time()

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

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)

        cv2.imshow('Video', frame)    
        # key= cv2.waitKey(0)
        if  cv2.waitKey(1) & 0xFF == ord('q'): # esc key
            break
    
    # check if the video writer is None
    if writer is None:
        # initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, fourcc, 30, (frame.shape[1], frame.shape[0]), True )

        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

    # write output frame to disk
    writer.write(frame)
print("[INFO] cleaning up...")
writer.release()
cap.release()
cv2.destroyAllWindows()