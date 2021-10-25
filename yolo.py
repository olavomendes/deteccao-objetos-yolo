import cv2
import numpy as np

cap = cv2.VideoCapture(0)
weight_height_target = 320
conf_threshold = 0.5
nms_threshold = 0.3

classes_file = 'Detecção de Objetos com YOLO/coco.names'
classes_names = []

with open(classes_file, 'rt') as file:
    classes_names = file.read().rstrip('\n').split('\n')

# print(classes_names)

model_config = 'Detecção de Objetos com YOLO/yolo/yolov3.cfg'
model_weights = 'Detecção de Objetos com YOLO/yolo/yolov3.weights'

net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

 
def find_objects(outputs, img):
    height_t, weight_t, confidence_t = img.shape
    bbox = []
    classes_ids = []
    conf_values = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                w, h = int(detection[2] * weight_t), int(detection[3] * height_t)
                x, y = int((detection[0] * weight_t) - w / 2), int((detection[1] * height_t) - h / 2)
                bbox.append([x, y, w, h])
                classes_ids.append(class_id)
                conf_values.append(float(confidence))
    
    #print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, conf_values, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0] # Remove colchetes extras
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(img, f'{classes_names[classes_ids[i]].upper()} {int(conf_values[i] * 100)}%',
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

while True:
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img, 1/255, (weight_height_target, weight_height_target), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    # print(layer_names)
    # print(net.getUnconnectedOutLayers())
    output_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()] # index
    # print(output_names)

    outputs = net.forward(output_names) # bb

    find_objects(outputs, img)
    
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()