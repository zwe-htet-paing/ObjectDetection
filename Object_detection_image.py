######################## READ IMAGE ############################
# import cv2
# # LOAD AN IMAGE USING 'IMREAD'
# img = cv2.imread("Resources/lena.png")
# # DISPLAY
# cv2.imshow("Lena Soderberg",img)
# cv2.waitKey(0)

import cv2
import numpy as np
import yaml

# loading paths from config.yaml
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# print(config)
img_path = config['img_path']
weight_path = config['yolo']['weight_path']
cfg_path = config['yolo']['cfg_path']
class_path = config['yolo']['class_path']

print(img_path)
print(weight_path)
print(cfg_path)
print(class_path)

# loading yolo network
net = cv2.dnn.readNet(weight_path, cfg_path)

# loading classes
classes = []
with open(class_path, 'r') as f:
    classes = f.read().splitlines()

# loading image

image = cv2.imread(img_path)
filename = img_path.split('/')[-1]

height, width, _ = image.shape
# print(height, width)

blob = cv2.dnn.blobFromImage(image, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)

net.setInput(blob)
output_layers_name = net.getUnconnectedOutLayersNames()
# print(output_layers_name)
layer_outputs = net.forward(output_layers_name)
# print(layers_output)

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
    confidence = float(round(confidences[i], 2))
    color = colors[i]

    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
    # cv2.putText(image, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
    text = "{}: {:.4f}".format(label, confidence)
    cv2.putText(image, text, (x, y -5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cv2.imshow('Image', image) 
cv2.imwrite('output/'+ filename, image)   
key = cv2.waitKey(0)

cv2.destroyAllWindows()
