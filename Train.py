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
import random

# === CONFIG ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-5  # Ridotto!
WEIGHT_DECAY = 1e-4   # Aggiunta regolarizzazione

# === DATASET CON DEBUG ===
class BBoxDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, debug=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.debug = debug
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg") or f.endswith(".png")]
        
        if self.debug:
            print(f"🔍 DEBUG: Trovate {len(self.img_files)} immagini")
            self._debug_labels()

    def _debug_labels(self):
        """Analizza le prime etichette per debugging"""
        print("\n📊 ANALISI ETICHETTE (prime 5):")
        for i in range(min(5, len(self.img_files))):
            img_name = self.img_files[i]
            label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")
            
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    bbox = f.readline().strip().split()
                print(f"  {img_name}: {bbox}")
            else:
                print(f"  {img_name}: ETICHETTA MANCANTE")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")

        # Carica immagine
        image = Image.open(img_path).convert("RGB")
        original_size = image.size  # Per debugging
        
        if self.transform:
            image = self.transform(image)

        # Carica bbox
        with open(label_path, "r") as f:
            bbox_str = f.readline().strip().split()
            bbox = [float(x) for x in bbox_str]
        
        # IMPORTANTE: Verifica che le coordinate siano normalizzate
        if self.debug and idx < 3:
            print(f"🔍 Immagine {idx}: {img_name}")
            print(f"   Dimensioni originali: {original_size}")
            print(f"   BBox raw: {bbox}")
            print(f"   BBox range: x∈[{min(bbox[0], bbox[0]+bbox[2]):.3f}, {max(bbox[0], bbox[0]+bbox[2]):.3f}], "
                  f"y∈[{min(bbox[1], bbox[1]+bbox[3]):.3f}, {max(bbox[1], bbox[1]+bbox[3]):.3f}]")
        
        bbox = torch.tensor(bbox, dtype=torch.float32)
        return image, bbox

# === MODELLO CON INIZIALIZZAZIONE MIGLIORE ===
class ResNetBBox(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.base.fc.in_features
        
        # Inizializzazione migliore
        self.base.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 5)  # x_center, y_center, w, h, class_prob
        )
        
        # Inizializzazione Xavier
        for m in self.base.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.base(x)

# === LOSS FUNCTION MIGLIORATA ===
class BBoxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        # Loss per coordinate (x, y, w, h)
        coord_loss = self.mse(pred[:, :4], target[:, :4])
        
        # Loss L1 per stabilità
        l1_loss = self.l1(pred[:, :4], target[:, :4])
        
        # Loss per confidence (se presente)
        conf_loss = 0
        if pred.shape[1] > 4 and target.shape[1] > 4:
            conf_loss = self.mse(pred[:, 4], target[:, 4])
        
        return coord_loss + 0.1 * l1_loss + 0.1 * conf_loss

# === TRAINING FUNCTION MIGLIORATA ===
def train_model(img_dir, label_dir):
    # Data augmentation migliorata
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    # Dataset con debug
    #dataset = BBoxDataset(img_dir, label_dir, transform=transform, debug=True)
    # Crea il dataset completo
    full_dataset = BBoxDataset(img_dir, label_dir, transform=transform)
    # Imposta la dimensione del subset (es. 100 campioni)
    subset_size = 100
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    subset_indices = indices[:subset_size]

# Crea il subset
    dataset = Subset(full_dataset, subset_indices)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Modello
    model = ResNetBBox().to(device)
    
    # Loss e optimizer migliorati
    criterion = BBoxLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    print(f"\n🚀 Inizio training su {len(dataset)} immagini")
    print(f"📊 Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LEARNING_RATE}")

    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            preds = model(images)
            loss = criterion(preds, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.6f}'})
            
            # Debug ogni 1000 batch
            if batch_idx % 1000 == 0 and batch_idx > 0:
                print(f"\n🔍 Batch {batch_idx}: Loss = {loss.item():.6f}")
                print(f"   Pred sample: {preds[0].detach().cpu().numpy()}")
                print(f"   Target sample: {targets[0].detach().cpu().numpy()}")

        avg_loss = epoch_loss / num_batches
        scheduler.step(avg_loss)
        
        print(f"\n📊 Epoch {epoch+1}/{EPOCHS} - Avg Loss: {avg_loss:.6f}")
        print(f"🎯 Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Salva il miglior modello
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_resnet_bbox_model.pth")
            print(f"✅ Nuovo miglior modello salvato! Loss: {best_loss:.6f}")

    print("✅ Training completato!")

    # Salva modello finale
    torch.save(model.state_dict(), "final_resnet_bbox_model.pth")
    
    # Salva configurazione
    config = {
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "best_loss": best_loss
    }
    with open("training_config_v2.json", "w") as f:
        json.dump(config, f, indent=2)

# === MAIN ===
if __name__ == "__main__":
    img_dir = '/home/francesca/Scrivania/ResNetBBoxRegressor/Dataset/images/train'
    label_dir = '/home/francesca/Scrivania/ResNetBBoxRegressor/Dataset/labels/train'
    
    print("🔍 VERIFICA PRELIMINARE DEI DATI...")
    
    # Verifica che le directory esistano
    if not os.path.exists(img_dir):
        print(f"❌ Directory immagini non trovata: {img_dir}")
        exit(1)
    if not os.path.exists(label_dir):
        print(f"❌ Directory etichette non trovata: {label_dir}")
        exit(1)
    
    train_model(img_dir, label_dir)