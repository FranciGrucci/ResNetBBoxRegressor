import os
import numpy as np
from PIL import Image

# Questo script converte le coordinate dei landmark in formato YOLO per la pupilla

# Percorsi delle cartelle: immagini, landmark e cartella dove salverai le label in formato YOLO
image_folder = "/home/francesca/Scrivania/ResNetBBoxRegressor/Dataset/images"
landmark_folder = "/home/francesca/Scrivania/ResNetBBoxRegressor/Dataset/labels"
label_folder = "/home/francesca/Scrivania/ResNetBBoxRegressor/labels_yolo"  # qui salverai le label YOLO

# Creo la cartella per salvare le label YOLO se non esiste già
os.makedirs(label_folder, exist_ok=True)

# Dimensione fissa della bounding box in pixel (40x40) per la pupilla
box_size = 40

# Funzione per normalizzare la bbox in coordinate YOLO: valori tra 0 e 1 relativi alla dimensione immagine
def normalize_bbox(cx, cy, w, h, img_width, img_height):
    # Divide le coordinate e dimensioni per larghezza e altezza dell'immagine
    return cx/img_width, cy/img_height, w/img_width, h/img_height

# Ciclo su tutti i file della cartella dei landmark
for file_name in os.listdir(landmark_folder):
    # Considero solo i file .txt (che contengono i punti landmark)
    if not file_name.endswith('.txt'):
        continue

    # Associo il nome immagine corrispondente al file txt (cambio estensione in .jpg)
    image_name = file_name.replace('.txt', '.jpg')

    # Percorsi completi per immagine, landmark e file label YOLO da creare
    img_path = os.path.join(image_folder, image_name)
    landmarks_path = os.path.join(landmark_folder, file_name)
    label_path = os.path.join(label_folder, file_name)

    # Se l'immagine non esiste, salto questo ciclo
    if not os.path.exists(img_path):
        continue

    # Apro l'immagine e prendo larghezza e altezza
    img = Image.open(img_path)
    W, H = img.size

    # Leggo i landmark dal file txt: numpy carica un array Nx2 (coordinate x,y)
    landmarks = np.loadtxt(landmarks_path)

    # DEBUG: stampo shape e coordinate landmarks per controllare (e esco dal ciclo con exit)
    print(f"{file_name} -> landmarks shape: {landmarks.shape}")
    print(landmarks)
    exit()

    # --- Qui inizia la parte per calcolare il centro della pupilla ---
    # Prendo due punti dei landmark della pupilla (ad es. punti 37 e 40, cioè indici 36 e 39 in Python)
    x1, y1 = landmarks[36]
    x2, y2 = landmarks[39]

    # Centro della bounding box è la media tra i due punti
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # La larghezza e altezza della bbox è quella fissa definita sopra (box_size)
    w = h = box_size

    # Uso le coordinate centrate sulla pupilla
    x_center = cx
    y_center = cy

    # Normalizzo la bounding box con la funzione definita sopra, in coordinate relative all'immagine (valori tra 0 e 1)
    x_yolo, y_yolo, w_yolo, h_yolo = normalize_bbox(x_center, y_center, w, h, W, H)

    # Scrivo il file label in formato YOLO (classe 0 + bbox normalizzata)
    # La classe 0 indica "pupilla"
    with open(label_path, 'w') as f:
        f.write(f"0 {x_yolo:.6f} {y_yolo:.6f} {w_yolo:.6f} {h_yolo:.6f}\n")
