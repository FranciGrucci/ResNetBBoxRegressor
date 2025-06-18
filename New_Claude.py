import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import random
import numpy as np
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

# === CONFIG ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50  # Aumentato per permettere convergenza graduale
LEARNING_RATE = 1e-4  # Aumentato leggermente
WEIGHT_DECAY = 1e-5   # Ridotto per non limitare troppo l'apprendimento

# === ANALISI DATASET ===
def analyze_dataset_distribution(dataset):
    """Analizza la distribuzione delle bounding box nel dataset"""
    print("\n📊 ANALIZZANDO DISTRIBUZIONE DEL DATASET...")
    
    centers_x = []
    centers_y = []
    widths = []
    heights = []
    
    # Prendi un campione rappresentativo
    sample_size = min(500, len(dataset))
    indices = random.sample(range(len(dataset)), sample_size)
    
    for i in tqdm(indices, desc="Analizzando campioni"):
        try:
            _, bbox = dataset[i]
            if isinstance(bbox, torch.Tensor):
                bbox = bbox.numpy()
            
            x, y, w, h = bbox[:4]  # Primi 4 elementi sono sempre x, y, w, h
            centers_x.append(x)
            centers_y.append(y)
            widths.append(w)
            heights.append(h)
        except Exception as e:
            print(f"Errore nel campione {i}: {e}")
            continue
    
    if len(centers_x) == 0:
        print("❌ Nessun dato valido trovato!")
        return
    
    # Statistiche
    print(f"\n📈 STATISTICHE SU {len(centers_x)} CAMPIONI:")
    print(f"Center X: min={min(centers_x):.3f}, max={max(centers_x):.3f}, mean={np.mean(centers_x):.3f}, std={np.std(centers_x):.3f}")
    print(f"Center Y: min={min(centers_y):.3f}, max={max(centers_y):.3f}, mean={np.mean(centers_y):.3f}, std={np.std(centers_y):.3f}")
    print(f"Width: min={min(widths):.3f}, max={max(widths):.3f}, mean={np.mean(widths):.3f}, std={np.std(widths):.3f}")
    print(f"Height: min={min(heights):.3f}, max={max(heights):.3f}, mean={np.mean(heights):.3f}, std={np.std(heights):.3f}")
    
    # Crea grafici
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0,0].hist(centers_x, bins=50, alpha=0.7, color='blue')
        axes[0,0].set_title('Distribuzione Center X')
        axes[0,0].set_xlabel('Center X')
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].hist(centers_y, bins=50, alpha=0.7, color='green')
        axes[0,1].set_title('Distribuzione Center Y')
        axes[0,1].set_xlabel('Center Y')
        axes[0,1].grid(True, alpha=0.3)
        
        axes[1,0].hist(widths, bins=50, alpha=0.7, color='red')
        axes[1,0].set_title('Distribuzione Width')
        axes[1,0].set_xlabel('Width')
        axes[1,0].grid(True, alpha=0.3)
        
        axes[1,1].hist(heights, bins=50, alpha=0.7, color='orange')
        axes[1,1].set_title('Distribuzione Height')
        axes[1,1].set_xlabel('Height')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dataset_distribution_analysis.png', dpi=150, bbox_inches='tight')
        print("📊 Grafici salvati in 'dataset_distribution_analysis.png'")
        plt.close()
    except Exception as e:
        print(f"⚠️ Errore nella creazione dei grafici: {e}")

# === DATA AUGMENTATION AVANZATA ===
class GazeAugmentation:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        
    def __call__(self, image, bbox):
        # Flip orizzontale (attenzione: aggiusta anche la bbox)
        if torch.rand(1) < 0.5:
            image, bbox = self.horizontal_flip(image, bbox)
        
        # Variazioni di colore
        if torch.rand(1) < 0.4:
            image = self.color_jitter(image)
        
        # Blur occasionale
        if torch.rand(1) < 0.2:
            image = self.gaussian_blur(image)
            
        # Rumore
        if torch.rand(1) < 0.1:
            image = self.add_noise(image)
        
        return image, bbox
    
    def horizontal_flip(self, image, bbox):
        """Flip orizzontale con correzione bbox"""
        image = F.hflip(image)
        
        # Correzione bbox: se le coordinate sono normalizzate [0,1]
        # x_nuovo = 1.0 - x_vecchio
        if len(bbox) >= 4:
            bbox = bbox.clone() if isinstance(bbox, torch.Tensor) else torch.tensor(bbox)
            bbox[0] = 1.0 - bbox[0]  # Inverte x_center
        
        return image, bbox
    
    def color_jitter(self, image):
        """Variazioni di colore per robustezza"""
        transform = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        )
        return transform(image)
    
    def gaussian_blur(self, image):
        """Blur per simulare condizioni reali"""
        return F.gaussian_blur(image, kernel_size=3, sigma=(0.1, 1.0))
    
    def add_noise(self, image):
        """Aggiunge rumore gaussiano"""
        if isinstance(image, torch.Tensor):
            noise = torch.randn_like(image) * 0.02
            image = torch.clamp(image + noise, 0, 1)
        return image

# === DATASET MIGLIORATO ===
class ImprovedBBoxDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, augmentation=None, debug=False, split='train'):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.augmentation = augmentation
        self.debug = debug
        self.split = split
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
        
        # Normalizzazione ImageNet
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        if self.debug:
            print(f"🔍 DEBUG: Trovate {len(self.img_files)} immagini in modalità {split}")
            self._debug_labels()

    def _debug_labels(self):
        """Analizza le prime etichette per debugging"""
        print("\n📊 ANALISI ETICHETTE (prime 5):")
        valid_labels = 0
        
        for i in range(min(10, len(self.img_files))):
            img_name = self.img_files[i]
            label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")
            
            if os.path.exists(label_path):
                try:
                    with open(label_path, "r") as f:
                        bbox = f.readline().strip().split()
                    bbox_floats = [float(x) for x in bbox]
                    print(f"  {img_name}: {bbox_floats}")
                    valid_labels += 1
                except Exception as e:
                    print(f"  {img_name}: ERRORE LETTURA - {e}")
            else:
                print(f"  {img_name}: ETICHETTA MANCANTE")
        
        print(f"✅ Etichette valide: {valid_labels}/{min(10, len(self.img_files))}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")

        try:
            # Carica immagine
            image = Image.open(img_path).convert("RGB")
            original_size = image.size

            # Carica bbox
            with open(label_path, "r") as f:
                bbox_str = f.readline().strip().split()
                bbox = [float(x) for x in bbox_str]
            
            # Assicurati che ci siano almeno 4 valori
            if len(bbox) < 4:
                bbox = bbox + [0.0] * (4 - len(bbox))
            
            # Se c'è solo 4 valori, aggiungi confidence = 1.0
            if len(bbox) == 4:
                bbox.append(1.0)
            
            bbox = torch.tensor(bbox, dtype=torch.float32)
            
            # Converti immagine in tensor per augmentation
            if not isinstance(image, torch.Tensor):
                image = transforms.ToTensor()(image)
            
            # Applica augmentation solo durante training
            if self.split == 'train' and self.augmentation:
                image, bbox = self.augmentation(image, bbox)
            
            # Resize se necessario
            if image.shape[1] != 224 or image.shape[2] != 224:
                image = F.resize(image, (224, 224))
            
            # Normalizza
            image = self.normalize(image)
            
            # Debug per i primi campioni
            if self.debug and idx < 3:
                print(f"🔍 Campione {idx}: {img_name}")
                print(f"   Dimensioni originali: {original_size}")
                print(f"   BBox: {bbox.numpy()}")
                print(f"   Range bbox: x∈[{bbox[0]-.5*bbox[2]:.3f}, {bbox[0]+.5*bbox[2]:.3f}], "
                      f"y∈[{bbox[1]-.5*bbox[3]:.3f}, {bbox[1]+.5*bbox[3]:.3f}]")
            
            return image, bbox
            
        except Exception as e:
            print(f"❌ Errore nel caricamento del campione {idx} ({img_name}): {e}")
            # Ritorna un campione di fallback
            dummy_image = torch.zeros(3, 224, 224)
            dummy_bbox = torch.tensor([0.5, 0.5, 0.1, 0.1, 0.0], dtype=torch.float32)
            return dummy_image, dummy_bbox

# === MODELLO MIGLIORATO ===
class ImprovedResNetBBox(nn.Module):
    def __init__(self, num_outputs=5):
        super().__init__()
        
        # Backbone più potente
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Rimuovi classifier finale
        self.backbone.fc = nn.Identity()
        
        # Regression head più sofisticato
        self.regression_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_outputs)  # x, y, w, h, confidence
        )
        
        # Inizializzazione corretta
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Inizializzazione dei pesi per training stabile"""
        for m in self.regression_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Estrai features
        features = self.backbone(x)
        
        # Regressione
        output = self.regression_head(features)
        
        return output

# === LOSS FUNCTION AVANZATA ===
class AdvancedBBoxLoss(nn.Module):
    def __init__(self, coord_weight=1.0, conf_weight=0.1):
        super().__init__()
        self.coord_weight = coord_weight
        self.conf_weight = conf_weight
        self.smooth_l1 = nn.SmoothL1Loss()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        # Separate coordinate and confidence predictions
        pred_coords = pred[:, :4]  # x, y, w, h
        target_coords = target[:, :4]
        
        # Coordinate loss (Smooth L1 è meglio per bounding box)
        coord_loss = self.smooth_l1(pred_coords, target_coords)
        
        # Confidence loss (se presente)
        conf_loss = 0
        if pred.shape[1] > 4 and target.shape[1] > 4:
            pred_conf = pred[:, 4]
            target_conf = target[:, 4]
            
            # Calcola IoU come target per confidence
            with torch.no_grad():
                iou = self.calculate_iou_batch(pred_coords, target_coords)
                # Usa IoU come target confidence se non è fornita
                if torch.all(target_conf == 1.0) or torch.all(target_conf == 0.0):
                    target_conf = iou
            
            conf_loss = self.mse(torch.sigmoid(pred_conf), target_conf)
        
        # Loss totale
        total_loss = self.coord_weight * coord_loss + self.conf_weight * conf_loss
        
        return total_loss, coord_loss, conf_loss
    
    def calculate_iou_batch(self, pred_bbox, target_bbox):
        """Calcola IoU per un batch di bounding box"""
        # Converti da center-size a corner format
        pred_corners = self.center_to_corners(pred_bbox)
        target_corners = self.center_to_corners(target_bbox)
        
        # Calcola IoU
        iou = self.box_iou(pred_corners, target_corners)
        
        return iou.clamp(0, 1)
    
    def center_to_corners(self, bbox):
        """Converti da formato center-size a corner format"""
        x_center, y_center, width, height = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    def box_iou(self, boxes1, boxes2):
        """Calcola IoU tra due set di bounding box"""
        # Calcola area di intersezione
        x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
        
        intersection = torch.clamp(x2 - x1, 0) * torch.clamp(y2 - y1, 0)
        
        # Calcola area delle box
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        union = area1 + area2 - intersection
        
        iou = intersection / (union + 1e-6)
        
        return iou

# === TRAINING FUNCTION AVANZATA ===
def train_model_advanced(img_dir, label_dir):
    print("🚀 INIZIALIZZAZIONE TRAINING AVANZATO...")
    
    # Augmentation
    augmentation = GazeAugmentation()
    
    # Dataset completo per analisi
    print("📊 Creazione dataset per analisi...")
    analysis_dataset = ImprovedBBoxDataset(
        img_dir, label_dir, 
        transform=None, 
        augmentation=None, 
        debug=False,
        split='analysis'
    )
    
    # Analizza distribuzione
    analyze_dataset_distribution(analysis_dataset)
    
    # Datasets per training
    print("🔧 Creazione datasets per training...")
    full_dataset = ImprovedBBoxDataset(
        img_dir, label_dir, 
        transform=None, 
        augmentation=None, 
        debug=True,
        split='train'
    )
    
    # Split train/validation
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    indices = list(range(dataset_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Training dataset con augmentation
    train_dataset = ImprovedBBoxDataset(
        img_dir, label_dir, 
        transform=None, 
        augmentation=augmentation, 
        debug=False,
        split='train'
    )
    train_dataset = Subset(train_dataset, train_indices)
    
    # Validation dataset senza augmentation
    val_dataset = ImprovedBBoxDataset(
        img_dir, label_dir, 
        transform=None, 
        augmentation=None, 
        debug=False,
        split='val'
    )
    val_dataset = Subset(val_dataset, val_indices)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"📊 Dataset split: Train={len(train_dataset)}, Validation={len(val_dataset)}")
    
    # Modello
    model = ImprovedResNetBBox().to(device)
    
    # Loss e optimizer
    criterion = AdvancedBBoxLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Scheduler più sofisticato
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    print(f"\n🚀 Inizio training avanzato")
    print(f"📊 Modello: {sum(p.numel() for p in model.parameters())} parametri")
    print(f"📊 Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LEARNING_RATE}")
    
    # Tracking delle metriche
    train_losses = []
    val_losses = []
    train_coord_losses = []
    train_conf_losses = []
    val_coord_losses = []
    val_conf_losses = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    for epoch in range(EPOCHS):
        # === TRAINING ===
        model.train()
        epoch_train_loss = 0
        epoch_train_coord_loss = 0
        epoch_train_conf_loss = 0
        num_train_batches = 0
        
        train_progress = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, (images, targets) in enumerate(train_progress):
            try:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                # Forward pass
                preds = model(images)
                total_loss, coord_loss, conf_loss = criterion(preds, targets)

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()

                # Accumula losses
                epoch_train_loss += total_loss.item()
                epoch_train_coord_loss += coord_loss.item()
                epoch_train_conf_loss += conf_loss.item() if isinstance(conf_loss, torch.Tensor) else conf_loss
                num_train_batches += 1
                
                # Update progress bar
                train_progress.set_postfix({
                    'Total': f'{total_loss.item():.6f}',
                    'Coord': f'{coord_loss.item():.6f}',
                    'Conf': f'{conf_loss.item() if isinstance(conf_loss, torch.Tensor) else conf_loss:.6f}'
                })
                
                # Debug periodico
                if batch_idx % 50 == 0 and batch_idx > 0:
                    with torch.no_grad():
                        pred_sample = preds[0].detach().cpu().numpy()
                        target_sample = targets[0].detach().cpu().numpy()
                        print(f"\n🔍 Batch {batch_idx}:")
                        print(f"   Pred: [{pred_sample[0]:.3f}, {pred_sample[1]:.3f}, {pred_sample[2]:.3f}, {pred_sample[3]:.3f}]")
                        print(f"   Target: [{target_sample[0]:.3f}, {target_sample[1]:.3f}, {target_sample[2]:.3f}, {target_sample[3]:.3f}]")
                        
            except Exception as e:
                print(f"❌ Errore nel batch {batch_idx}: {e}")
                continue
        
        # === VALIDATION ===
        model.eval()
        epoch_val_loss = 0
        epoch_val_coord_loss = 0
        epoch_val_conf_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{EPOCHS}")
            
            for images, targets in val_progress:
                try:
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    
                    preds = model(images)
                    total_loss, coord_loss, conf_loss = criterion(preds, targets)
                    
                    epoch_val_loss += total_loss.item()
                    epoch_val_coord_loss += coord_loss.item()
                    epoch_val_conf_loss += conf_loss.item() if isinstance(conf_loss, torch.Tensor) else conf_loss
                    num_val_batches += 1
                    
                    val_progress.set_postfix({
                        'Total': f'{total_loss.item():.6f}',
                        'Coord': f'{coord_loss.item():.6f}',
                        'Conf': f'{conf_loss.item() if isinstance(conf_loss, torch.Tensor) else conf_loss:.6f}'
                    })
                    
                except Exception as e:
                    print(f"❌ Errore nella validazione: {e}")
                    continue
        
        # Calcola medie
        avg_train_loss = epoch_train_loss / max(num_train_batches, 1)
        avg_val_loss = epoch_val_loss / max(num_val_batches, 1)
        avg_train_coord_loss = epoch_train_coord_loss / max(num_train_batches, 1)
        avg_val_coord_loss = epoch_val_coord_loss / max(num_val_batches, 1)
        avg_train_conf_loss = epoch_train_conf_loss / max(num_train_batches, 1)
        avg_val_conf_loss = epoch_val_conf_loss / max(num_val_batches, 1)
        
        # Salva metriche
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_coord_losses.append(avg_train_coord_loss)
        val_coord_losses.append(avg_val_coord_loss)
        train_conf_losses.append(avg_train_conf_loss)
        val_conf_losses.append(avg_val_conf_loss)
        
        # Scheduler step
        scheduler.step()
        
        # Logging
        print(f"\n📊 Epoch {epoch+1}/{EPOCHS} Results:")
        print(f"   Train Loss: {avg_train_loss:.6f} (Coord: {avg_train_coord_loss:.6f}, Conf: {avg_train_conf_loss:.6f})")
        print(f"   Val Loss: {avg_val_loss:.6f} (Coord: {avg_val_coord_loss:.6f}, Conf: {avg_val_conf_loss:.6f})")
        print(f"   Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
        
        # Salva miglior modello
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': {
                    'img_size': IMG_SIZE,
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE,
                    'weight_decay': WEIGHT_DECAY
                }
            }, 'best_improved_gaze_model.pth')
            patience_counter = 0
            print(f"🎯 Nuovo miglior modello salvato! Val Loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"⚠️  Early stopping dopo {patience} epochs senza miglioramenti")
            break
    
    print("✅ Training completato!")
    
    # Salva modello finale
    torch.save(model.state_dict(), "final_improved_gaze_model.pth")
    
    # Salva metriche
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_coord_losses': train_coord_losses,
        'val_coord_losses': val_coord_losses,
        'train_conf_losses': train_conf_losses,
        'val_conf_losses': val_conf_losses,
        'best_val_loss': best_val_loss,
        'config': {
            'img_size': IMG_SIZE,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY
        }
    }
    
    with open("improved_training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Crea grafico delle loss
    try:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses, label='Val Loss', color='red')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(train_coord_losses, label='Train Coord Loss', color='blue')
        plt.plot(val_coord_losses, label='Val Coord Loss', color='red')
        plt.title('Coordinate Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.plot(train_conf_losses, label='Train Conf Loss', color='blue')
        plt.plot(val_conf_losses, label='Val Conf Loss', color='red')
        plt.title('Confidence Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('improved_training_curves.png', dpi=150, bbox_inches='tight')
        print("📈 Grafici di training salvati in 'improved_training_curves.png'")
        plt.close()
    except Exception as e:
        print(f"⚠️ Errore nella creazione dei grafici: {e}")
    
    return model

# === INFERENCE E TESTING ===
def test_model_performance(model, test_loader, criterion):
    """Testa le performance del modello"""
    print("\n🧪 TESTING PERFORMANCE DEL MODELLO...")
    
    model.eval()
    total_loss = 0
    total_coord_loss = 0
    total_conf_loss = 0
    num_samples = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for images, batch_targets in tqdm(test_loader, desc="Testing"):
            try:
                images = images.to(device)
                batch_targets = batch_targets.to(device)
                
                preds = model(images)
                loss, coord_loss, conf_loss = criterion(preds, batch_targets)
                
                total_loss += loss.item()
                total_coord_loss += coord_loss.item()
                total_conf_loss += conf_loss.item() if isinstance(conf_loss, torch.Tensor) else conf_loss
                num_samples += images.size(0)
                
                predictions.extend(preds.cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())
                
            except Exception as e:
                print(f"❌ Errore nel testing: {e}")
                continue
    
    avg_loss = total_loss / len(test_loader)
    avg_coord_loss = total_coord_loss / len(test_loader)
    avg_conf_loss = total_conf_loss / len(test_loader)
    
    print(f"📊 RISULTATI TEST:")
    print(f"   Total Loss: {avg_loss:.6f}")
    print(f"   Coordinate Loss: {avg_coord_loss:.6f}")
    print(f"   Confidence Loss: {avg_conf_loss:.6f}")
    print(f"   Samples: {num_samples}")
    
    # Calcola metriche aggiuntive
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Errore medio per coordinata
    coord_errors = np.abs(predictions[:, :4] - targets[:, :4])
    mean_coord_errors = np.mean(coord_errors, axis=0)
    
    print(f"\n📏 ERRORI MEDI PER COORDINATA:")
    print(f"   X Center: {mean_coord_errors[0]:.4f}")
    print(f"   Y Center: {mean_coord_errors[1]:.4f}")
    print(f"   Width: {mean_coord_errors[2]:.4f}")
    print(f"   Height: {mean_coord_errors[3]:.4f}")
    
    return {
        'avg_loss': avg_loss,
        'avg_coord_loss': avg_coord_loss,
        'avg_conf_loss': avg_conf_loss,
        'predictions': predictions,
        'targets': targets,
        'coord_errors': coord_errors,
        'mean_coord_errors': mean_coord_errors
    }

# === VISUALIZZAZIONE PREDIZIONI ===
def visualize_predictions(model, dataset, num_samples=8):
    """Visualizza predizioni del modello"""
    print(f"\n🎨 VISUALIZZAZIONE {num_samples} PREDIZIONI...")
    
    model.eval()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Prendi campioni casuali
    indices = random.sample(range(len(dataset)), num_samples)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            try:
                image, target = dataset[idx]
                
                # Prepara per inference
                if len(image.shape) == 3:
                    image_input = image.unsqueeze(0).to(device)
                else:
                    image_input = image.to(device)
                
                # Predizione
                pred = model(image_input)
                pred = pred.squeeze(0).cpu().numpy()
                target = target.cpu().numpy() if isinstance(target, torch.Tensor) else target
                
                # Prepara immagine per visualizzazione
                img_vis = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
                if img_vis.shape[0] == 3:  # CHW -> HWC
                    img_vis = np.transpose(img_vis, (1, 2, 0))
                
                # Denormalizza se necessario
                if img_vis.min() < 0:  # Probabilmente normalizzata
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_vis = img_vis * std + mean
                    img_vis = np.clip(img_vis, 0, 1)
                
                # Plot
                axes[i].imshow(img_vis)
                axes[i].set_title(f"Sample {idx}")
                axes[i].axis('off')
                
                # Disegna bounding boxes
                h, w = img_vis.shape[:2]
                
                # Target (verde)
                tx, ty, tw, th = target[:4]
                target_rect = plt.Rectangle(
                    ((tx - tw/2) * w, (ty - th/2) * h),
                    tw * w, th * h,
                    linewidth=2, edgecolor='green', facecolor='none', label='Target'
                )
                axes[i].add_patch(target_rect)
                
                # Predizione (rosso)
                px, py, pw, ph = pred[:4]
                pred_rect = plt.Rectangle(
                    ((px - pw/2) * w, (py - ph/2) * h),
                    pw * w, ph * h,
                    linewidth=2, edgecolor='red', facecolor='none', linestyle='--', label='Pred'
                )
                axes[i].add_patch(pred_rect)
                
                # Aggiungi testo con errori
                error_x = abs(px - tx)
                error_y = abs(py - ty)
                axes[i].text(5, 15, f'Ex:{error_x:.3f} Ey:{error_y:.3f}', 
                           color='white', fontsize=8, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
                
            except Exception as e:
                print(f"❌ Errore nella visualizzazione del campione {idx}: {e}")
                axes[i].text(0.5, 0.5, f'Error: {str(e)[:30]}...', 
                           transform=axes[i].transAxes, ha='center', va='center')
                axes[i].axis('off')
    
    # Legenda
    axes[0].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('improved_predictions_visualization.png', dpi=150, bbox_inches='tight')
    print("🎨 Visualizzazioni salvate in 'improved_predictions_visualization.png'")
    plt.close()

# === FUNZIONE DI INFERENCE PER NUOVE IMMAGINI ===
def predict_gaze_bbox(model, image_path, visualize=True):
    """Predice bounding box per una nuova immagine"""
    model.eval()
    
    try:
        # Carica e preprocessa immagine
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        
        # Transform per inference
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predizione
        with torch.no_grad():
            pred = model(image_tensor)
            pred = pred.squeeze(0).cpu().numpy()
        
        # Estrai coordinate
        x_center, y_center, width, height = pred[:4]
        confidence = pred[4] if len(pred) > 4 else 1.0
        
        print(f"🎯 PREDIZIONE per {image_path}:")
        print(f"   Center: ({x_center:.3f}, {y_center:.3f})")
        print(f"   Size: {width:.3f} x {height:.3f}")
        print(f"   Confidence: {confidence:.3f}")
        
        # Visualizzazione
        if visualize:
            plt.figure(figsize=(10, 8))
            plt.imshow(image)
            plt.title(f"Gaze Prediction - {os.path.basename(image_path)}")
            plt.axis('off')
            
            # Disegna bounding box
            w, h = original_size
            rect = plt.Rectangle(
                ((x_center - width/2) * w, (y_center - height/2) * h),
                width * w, height * h,
                linewidth=3, edgecolor='red', facecolor='none'
            )
            plt.gca().add_patch(rect)
            
            # Aggiungi punto centrale
            plt.plot(x_center * w, y_center * h, 'ro', markersize=8)
            
            # Testo con informazioni
            plt.text(10, 30, f'Confidence: {confidence:.3f}', 
                    color='white', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='red', alpha=0.8))
            
            plt.tight_layout()
            output_path = f"prediction_{os.path.splitext(os.path.basename(image_path))[0]}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"📸 Predizione salvata in {output_path}")
            plt.close()
        
        return {
            'bbox': [x_center, y_center, width, height],
            'confidence': confidence,
            'original_size': original_size
        }
        
    except Exception as e:
        print(f"❌ Errore nella predizione per {image_path}: {e}")
        return None

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("🚀 AVVIO TRAINING AVANZATO GAZE DETECTION")
    print("=" * 60)
    
    # Percorsi
    img_dir = './Dataset/images/train'
 
    label_dir = './Dataset/labels/train' # Cartella con le etichette .txt
    
    # Verifica cartelle
    if not os.path.exists(img_dir):
        print(f"❌ Cartella immagini non trovata: {img_dir}")
        print("📁 Crea la cartella 'images' e inserisci le immagini")
        exit(1)
    
    if not os.path.exists(label_dir):
        print(f"❌ Cartella etichette non trovata: {label_dir}")
        print("📁 Crea la cartella 'labels' e inserisci i file .txt con le coordinate")
        exit(1)
    
    print(f"✅ Cartelle trovate:")
    print(f"   Immagini: {img_dir}")
    print(f"   Etichette: {label_dir}")
    
    # Conta file
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    
    print(f"📊 File trovati:")
    print(f"   Immagini: {len(img_files)}")
    print(f"   Etichette: {len(label_files)}")
    
    if len(img_files) == 0:
        print("❌ Nessuna immagine trovata!")
        exit(1)
    
    if len(label_files) == 0:
        print("❌ Nessuna etichetta trovata!")
        exit(1)
    
    print(f"\n⚙️  CONFIGURAZIONE:")
    print(f"   Device: {device}")
    print(f"   Image Size: {IMG_SIZE}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Weight Decay: {WEIGHT_DECAY}")
    
    # Avvia training
    try:
        model = train_model_advanced(img_dir, label_dir)
        print("\n✅ TRAINING COMPLETATO CON SUCCESSO!")
        
        # Test su alcune immagini se disponibili
        test_images = img_files[:5]  # Prime 5 immagini per test
        
        print(f"\n🧪 TESTING SU {len(test_images)} IMMAGINI DI ESEMPIO...")
        for img_file in test_images:
            img_path = os.path.join(img_dir, img_file)
            result = predict_gaze_bbox(model, img_path, visualize=True)
            if result:
                print(f"✅ Predizione completata per {img_file}")
        
        print("\n🎉 PROCESSO COMPLETATO!")
        print("📁 File generati:")
        print("   - best_improved_gaze_model.pth (miglior modello)")
        print("   - final_improved_gaze_model.pth (modello finale)")
        print("   - improved_training_metrics.json (metriche)")
        print("   - improved_training_curves.png (grafici training)")
        print("   - improved_predictions_visualization.png (predizioni)")
        print("   - dataset_distribution_analysis.png (analisi dataset)")
        print("   - prediction_*.png (predizioni individuali)")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrotto dall'utente")
    except Exception as e:
        print(f"\n❌ Errore durante il training: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 Fine esecuzione")