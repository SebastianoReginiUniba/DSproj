# Librerie
import cv2
import numpy as np
from scipy.spatial import distance as dist
from sklearn.cluster import DBSCAN

# Costanti
TARGET_CLASSES = None
MIN_CONFIDENCE = 0.6
NMS_THRESHOLD = 0.5

SIZE = (320, 320)
MIN_DISTANCE = 30

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# Funzione che carica YOLO, un algoritmo di object detection
def load_yolo(yolo_path=""):
	net = cv2.dnn.readNet(yolo_path+"yolov3.weights", yolo_path+"yolov3.cfg")
	classes = []
	with open(yolo_path+"coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]
	layers_names = net.getLayerNames()
	output_layers = [layers_names[i-1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers

# Funzione che utilizza YOLO per identificare i soggetti nei singoli frame
def detect_objects(img, net, outputLayers):			
    img = cv2.resize(img, SIZE)
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=SIZE)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_detection_metadata(outputs, height, width):
    # Inizializza le liste che conterranno le informazioni di ogni rilevamento
    boxes = []  # Lista dei rettangoli delimitatori
    centroids = []  # Lista dei centroidi
    confs = []  # Lista delle confidenze

    # Scorre tutti gli output prodotti dal modello
    for output in outputs:
        # Scorre tutti i rilevamenti presenti in ogni output
        for detect in output:
            # Estrae la classe con la probabilità maggiore dal vettore di probabilità
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]

            # Se la classe non è tra quelle di interesse, salta questo rilevamento
            if(TARGET_CLASSES!=None and classes[class_id] not in TARGET_CLASSES):
                continue
            
            # Se la confidenza supera la soglia minima, elabora il rilevamento
            if(conf > MIN_CONFIDENCE):
                # Calcola le coordinate del rettangolo delimitatore
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)

                # Aggiunge le informazioni del rilevamento alle rispettive liste
                boxes.append([x, y, w, h])
                centroids.append((center_x,center_y))
                confs.append(conf.astype(float))
    
    # Restituisce le liste contenenti le informazioni di tutti i rilevamenti
    return boxes, centroids, confs

# Funzione che calcola le distanze utilizzando la funzione della libreria scipy
def compute_distances(centroids):
    dist_matrix = dist.cdist(centroids, centroids)
    dist_matrix = dist_matrix + np.eye(dist_matrix.shape[0], dist_matrix.shape[1])*1000

    return dist_matrix

# Funzione che recupera i centroidi che non rispettano la distanza minima
def get_contact_indices(centroids):
    dist_matrix = compute_distances(centroids)
    indices = np.where(dist_matrix<MIN_DISTANCE)
    contact_indices = list(zip(indices[0], indices[1]))
    return contact_indices

# Funzione che elimina i rilevamenti multipli dello stesso centroide
def non_max_suppression(boxes, centroids, confidences, min_confidence, threshold):
    
    boxesMax = []
    centroidsMax = []
    boxesIds = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, threshold)

    for boxId in boxesIds:
        boxesMax.append(boxes[boxId])
        centroidsMax.append(centroids[boxId])
    
    return boxesMax, centroidsMax

# Funzione che effettua la clusterizzazione utilizzando l'algoritmo DBSCAN
def find_groups(img, centroids):
    db = DBSCAN(eps=30, min_samples=2).fit(centroids)
    labels = db.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    text = "Clusters presenti nel frame: %d" % n_clusters
    cv2.putText(img, text, (335, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 1)
    
    labels = list(filter(lambda x: x >= 0, labels))
    # conta il numero di punti in ogni cluster
    counts = np.bincount(labels)
    # elimina il conteggio del cluster "outlier"
    counts = counts[1:]
    # calcola le percentuali dei cluster
    percentages = counts / len(centroids) * 100
    
    # scrive le percentuali dei cluster sull'immagine
    for i, percentage in enumerate(percentages):
        print("Cluster %d: %.2f%%, con numero di elementi %d su un totale di %d soggetti" % (i, percentage, counts[i], len(centroids)))
    print("\n")

# Funzione che rigenera il frame aggiungendo gli elementi delle misurazioni eseguite
# sul rilevamento, sulla distanza minima e sulla clusterizzazione
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

# Esecuzione principale del codice ----------------------

net, classes, colors, output_layers = load_yolo()

cap = cv2.VideoCapture("fin.mp4")
ret, frame = cap.read()

if(not ret):
    print("Errore durante il caricamento del video")
    exit(0)

# Elaborazione del video frame per frame
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
    print(centroids)
    
    if(cv2.waitKey(1)==ord("q")):
        break

cap.release()
cv2.destroyAllWindows()
