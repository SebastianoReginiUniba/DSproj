import cv2
import numpy as np
from scipy.spatial import distance as dist
from sklearn.cluster import DBSCAN

TARGET_CLASSES = None
MIN_CONFIDENCE = 0.6
NMS_THRESHOLD = 0.5

SIZE = (320, 320)
MIN_DISTANCE = 30

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

def load_yolo(yolo_path=""):
	net = cv2.dnn.readNet(yolo_path+"yolov3.weights", yolo_path+"yolov3.cfg")
	classes = []
	with open(yolo_path+"coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]
	layers_names = net.getLayerNames()
	output_layers = [layers_names[i-1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers


def detect_objects(img, net, outputLayers):			
    img = cv2.resize(img, SIZE)
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=SIZE)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_detection_metadata(outputs, height, width):

    boxes = []
    centroids = []
    confs = []

    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]

            if(TARGET_CLASSES!=None and classes[class_id] not in TARGET_CLASSES):
                continue
            
            if(conf > MIN_CONFIDENCE):
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                centroids.append((center_x,center_y))
                confs.append(conf.astype(float))
    
    return boxes, centroids, confs

def compute_distances(centroids):
    dist_matrix = dist.cdist(centroids, centroids)
    dist_matrix = dist_matrix + np.eye(dist_matrix.shape[0], dist_matrix.shape[1])*1000

    return dist_matrix

def get_contact_indices(centroids):
    dist_matrix = compute_distances(centroids)
    indices = np.where(dist_matrix<MIN_DISTANCE)
    contact_indices = list(zip(indices[0], indices[1]))
    return contact_indices

def non_max_suppression(boxes, centroids, confidences, min_confidence, threshold):
    
    boxesMax = []
    centroidsMax = []
    boxesIds = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, threshold)

    for boxId in boxesIds:
        boxesMax.append(boxes[boxId])
        centroidsMax.append(centroids[boxId])
    
    return boxesMax, centroidsMax

def draw_results(img, centroids, contact_indices, confs=None, show_conf=True):

    centroids_drawn = set()

    if show_conf:
        for i, centroid in enumerate(centroids):
            text = "{:.0f}%".format(confs[i]*100)
            color = GREEN if i not in [c1 for c1, c2 in contact_indices] else RED
            cv2.putText(img, text, (centroid[0], centroid[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, color, 1)

    for c1, c2 in contact_indices:
        centroid1 = centroids[c1]
        centroid2 = centroids[c2]

        cv2.circle(img, centroid1, 2, RED, cv2.FILLED)
        cv2.circle(img, centroid2, 2, RED, cv2.FILLED)

        cv2.line(img, centroid1, centroid2, RED, thickness=1)

        centroids_drawn.add(centroid1)
        centroids_drawn.add(centroid2)

    centroids_to_draw = set(centroids) - centroids_drawn

    if(show_conf and confs==None):
        print("Missing confidence array")
        return

    for centroid in centroids_to_draw:

        cv2.circle(img, centroid, 2, GREEN, cv2.FILLED)
        
    return img

def find_groups(img, centroids):
    db = DBSCAN(eps=30, min_samples=2).fit(centroids)
    labels = db.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    text = "Clusters presenti nel frame: %d" % n_clusters
    cv2.putText(img, text, (335, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 1)

net, classes, colors, output_layers = load_yolo()

cap = cv2.VideoCapture("fin.mp4")
ret, frame = cap.read()

if(not ret):
    print("Errore durante il caricamento del video")
    exit(0)


while(cap.isOpened()):
    
    ret, frame = cap.read()
    if(not ret):
        print("Errore durante il caricamento del video")
        break

    blob, outputs = detect_objects(frame, net, output_layers)
    boxes, centroids, confs = get_detection_metadata(outputs, frame.shape[0], frame.shape[1])

    _, centroids = non_max_suppression(boxes, centroids, confs, MIN_CONFIDENCE, NMS_THRESHOLD)

    contact_indices = get_contact_indices(centroids)
    frame_out = draw_results(frame, centroids, contact_indices, confs=confs, show_conf=True)
    find_groups(frame, centroids)
    cv2.imshow("Immagine", frame_out)
    
    
    
    if(cv2.waitKey(1)==ord("q")):
        break


cap.release()
cv2.destroyAllWindows()