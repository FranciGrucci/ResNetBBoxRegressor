import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class BBoxDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        # Salvo la cartella delle immagini
        self.img_dir = img_dir
        
        # Salvo la cartella dei file di label (bounding box)
        self.label_dir = label_dir
        
        # Salvo le trasformazioni da applicare alle immagini (es. resize, normalizzazione)
        self.transform = transform
        
        # Creo la lista di tutti i file immagini nella cartella img_dir con estensione .jpg o .png
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
        
        # Ordino la lista alfabeticamente per mantenere corrispondenza con i file label
        self.img_files.sort()

    def __len__(self):
        # Ritorna la lunghezza del dataset, cioè il numero totale di immagini
        return len(self.img_files)

    def __getitem__(self, idx):
        # Costruisco il percorso dell'immagine corrispondente all'indice idx
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        
        # Costruisco il percorso del file label corrispondente all'immagine (stesso nome, estensione .txt)
        label_path = os.path.join(self.label_dir, os.path.splitext(self.img_files[idx])[0] + '.txt')

        # Apro l'immagine e la converto in RGB (assicuro che abbia 3 canali)
        image = Image.open(img_path).convert("RGB")
        
        # Se è stata specificata una trasformazione, la applico all'immagine
        if self.transform:
            image = self.transform(image)

        # Apro il file label, leggo la prima riga e la pulisco da spazi vuoti
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            
            # Splitto la riga in singoli valori float (es. x_center, y_center, width, height, class_prob)
            bbox = list(map(float, line.split()))

        # Controllo che ci siano esattamente 5 valori, altrimenti lancio un errore
        assert len(bbox) == 5, f"Expected 5 values in label file, got {len(bbox)}"

        # Converto la lista in un tensore PyTorch di tipo float32
        bbox = torch.tensor(bbox, dtype=torch.float32)
        
        # Ritorno una tupla contenente l'immagine e il tensore con i valori di bounding box
        return image, bbox
