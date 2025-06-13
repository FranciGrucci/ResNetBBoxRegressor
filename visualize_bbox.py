import os
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms, models
import numpy as np
from torch.utils.data import Subset
import random

# === CONFIGURAZIONE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

# === DEFINIZIONE MODELLO (uguale al training) ===
class ResNetBBox(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.base.fc.in_features
        
        # Architettura migliorata (uguale al training)
        self.base.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 5)  # x_center, y_center, w, h, class_prob
        )

    def forward(self, x):
        return self.base(x)

# === DATASET (uguale al training) ===
class BBoxDataset:
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg") or f.endswith(".png")]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Carica ground truth se esiste
        bbox_gt = None
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                bbox_gt = list(map(float, f.readline().split()))
            bbox_gt = torch.tensor(bbox_gt, dtype=torch.float32)

        return image, bbox_gt, img_name

# === FUNZIONI UTILITY ===
def denormalize_bbox(bbox, img_width, img_height):
    """Converte bbox normalizzate in coordinate pixel"""
    x_center, y_center, width, height = bbox[:4]
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    return x_center, y_center, width, height

def center_to_corners(x_center, y_center, width, height):
    """Converte da formato center a formato corners (x1, y1, x2, y2)"""
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return x1, y1, x2, y2

def compute_iou(box1, box2):
    """Calcola IoU tra due bounding box"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0

    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

# === FUNZIONE PRINCIPALE ===
def visualize_predictions():
    # Percorsi
    img_dir = "/home/francesca/Scrivania/ResNetBBoxRegressor/Dataset/images/test"
    label_dir = "/home/francesca/Scrivania/ResNetBBoxRegressor/Dataset/labels/test"
    model_path = "best_resnet_bbox_model.pth"  # Usa il modello migliore!
    
    # Controlla se i percorsi esistono
    if not os.path.exists(img_dir):
        print(f"❌ Directory immagini non trovata: {img_dir}")
        return
    if not os.path.exists(model_path):
        print(f"❌ Modello non trovato: {model_path}")
        return
    
    # Trasformazioni (uguali al training)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Carica dataset
    #dataset = BBoxDataset(img_dir, label_dir, transform=transform)
    full_dataset = BBoxDataset(img_dir, label_dir, transform=transform)
    # Imposta la dimensione del subset (es. 100 campioni)
    subset_size = 100
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    subset_indices = indices[:subset_size]

# Crea il subset
    dataset = Subset(full_dataset, subset_indices)
    print(f"📁 Trovate {len(dataset)} immagini nel dataset")
    
    # Carica modello
    model = ResNetBBox().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("✅ Modello caricato con successo")
    except Exception as e:
        print(f"❌ Errore nel caricamento del modello: {e}")
        return
    
    # Crea directory per i risultati
    output_dir = "visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Processo alcune immagini
    num_images_to_process = min(5, len(dataset))
    total_iou = 0
    valid_comparisons = 0
    
    print(f"\n🔍 Elaborando {num_images_to_process} immagini...")
    
    for i in range(num_images_to_process):
        # Carica immagine e ground truth
        img_tensor, bbox_gt, img_name = dataset[i]
        
        # Carica immagine originale (non trasformata)
        img_path = os.path.join(img_dir, img_name)
        original_img = Image.open(img_path).convert("RGB")
        W, H = original_img.size
        
        # Predizione
        with torch.no_grad():
            img_batch = img_tensor.unsqueeze(0).to(device)
            pred = model(img_batch)
            pred_bbox = pred.squeeze().cpu().numpy()
        
        # Crea copia per il disegno
        draw_img = original_img.copy()
        draw = ImageDraw.Draw(draw_img)
        
        # Font per il testo (usa default se non disponibile)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Disegna predizione
        if len(pred_bbox) >= 4:
            pred_x, pred_y, pred_w, pred_h = denormalize_bbox(pred_bbox, W, H)
            pred_x1, pred_y1, pred_x2, pred_y2 = center_to_corners(pred_x, pred_y, pred_w, pred_h)
            
            # Disegna bounding box predetta
            draw.rectangle([pred_x1, pred_y1, pred_x2, pred_y2], outline='red', width=3)
            draw.text((pred_x1, pred_y1-20), "PRED", fill='red', font=font)
            
            print(f"🔸 Immagine {i+1}: {img_name}")
            print(f"   Predizione: center=({pred_x:.1f}, {pred_y:.1f}), size=({pred_w:.1f}x{pred_h:.1f})")
            
            # Confidence score se disponibile
            if len(pred_bbox) >= 5:
                confidence = pred_bbox[4]
                draw.text((pred_x1, pred_y1-40), f"Conf: {confidence:.3f}", fill='red', font=font)
                print(f"   Confidence: {confidence:.3f}")
        
        # Disegna ground truth se disponibile
        if bbox_gt is not None and len(bbox_gt) >= 4:
            gt_x, gt_y, gt_w, gt_h = denormalize_bbox(bbox_gt.numpy(), W, H)
            gt_x1, gt_y1, gt_x2, gt_y2 = center_to_corners(gt_x, gt_y, gt_w, gt_h)
            
            # Disegna bounding box ground truth
            draw.rectangle([gt_x1, gt_y1, gt_x2, gt_y2], outline='green', width=3)
            draw.text((gt_x1, gt_y1-20), "GT", fill='green', font=font)
            
            print(f"   Ground Truth: center=({gt_x:.1f}, {gt_y:.1f}), size=({gt_w:.1f}x{gt_h:.1f})")
            
            # Calcola IoU
            if len(pred_bbox) >= 4:
                iou = compute_iou([pred_x1, pred_y1, pred_x2, pred_y2], 
                                [gt_x1, gt_y1, gt_x2, gt_y2])
                print(f"   IoU: {iou:.3f}")
                total_iou += iou
                valid_comparisons += 1
                
                # Mostra IoU sull'immagine
                draw.text((10, 10), f"IoU: {iou:.3f}", fill='blue', font=font)
        
        # Salva immagine
        output_path = os.path.join(output_dir, f"result_{i+1}_{img_name}")
        draw_img.save(output_path)
        print(f"   ✅ Salvata: {output_path}")
        print()
    
    # Statistiche finali
    if valid_comparisons > 0:
        avg_iou = total_iou / valid_comparisons
        print(f"📊 IoU medio: {avg_iou:.3f} su {valid_comparisons} immagini")
    
    print(f"🎯 Visualizzazione completata! Risultati salvati in: {output_dir}/")

# === ESECUZIONE ===
if __name__ == "__main__":
    visualize_predictions()