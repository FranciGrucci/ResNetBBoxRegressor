import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Subset
import random

# === CONFIG ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4

# === DATASET ===
class BBoxDataset(Dataset):
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

        with open(label_path, "r") as f:
            bbox = list(map(float, f.readline().split()))
        bbox = torch.tensor(bbox, dtype=torch.float32)

        return image, bbox

# === MODEL ===
class ResNetBBox(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.base.fc.in_features
        self.base.fc = nn.Linear(in_features, 5)  # x_center, y_center, w, h, class_prob

    def forward(self, x):
        return self.base(x)

# === METRIC ===
def mean_average_precision(preds, targets, iou_threshold=0.5):
    def compute_iou(box1, box2):
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2

        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)

        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area != 0 else 0

    correct = 0
    total = len(preds)

    for pred, target in zip(preds, targets):
        pred = pred.detach().cpu().tolist()
        target = target.detach().cpu().tolist()
        iou = compute_iou(pred, target)
        if iou >= iou_threshold:
            correct += 1

    return correct / total if total > 0 else 0

# === TRAIN FUNCTION ===
def train_model(img_dir, label_dir):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    #dataset = BBoxDataset(img_dir, label_dir, transform=transform)
    # Crea il dataset completo
    #full_dataset = BBoxDataset(img_dir, label_dir, transform=transform)
    # Imposta la dimensione del subset (es. 100 campioni)
    #subset_size = 100
    #indices = list(range(len(full_dataset)))
    #random.shuffle(indices)
    #subset_indices = indices[:subset_size]

# Crea il subset
    #dataset = Subset(full_dataset, subset_indices)
    dataset = BBoxDataset(img_dir, label_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ResNetBBox().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    all_preds = []
    all_targets = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for images, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = images.to(device)
            targets = targets.to(device)

            preds = model(images)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            all_preds.extend(preds.detach())
            all_targets.extend(targets.detach())

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss / len(dataloader):.4f}")

    print("✅ Addestramento completato!")

    # === SAVE MODEL ===
    torch.save(model.state_dict(), "resnet_bbox_model.pth")
    print("✅ Pesi salvati in resnet_bbox_model.pth")

    # === SAVE CONFIG ===
    config = {
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE
    }
    with open("training_config.json", "w") as f:
        json.dump(config, f)
    print("✅ Config salvata in training_config.json")

    # === EVAL mAP ===
    map_score = mean_average_precision(all_preds, all_targets)
    print(f"📊 mAP @IoU=0.5: {map_score:.4f}")

# === MAIN ===
if __name__ == "__main__":
    img_dir = '/home/francesca/Scrivania/ResNetBBoxRegressor/Dataset/images/train'
    label_dir = '/home/francesca/Scrivania/ResNetBBoxRegressor/Dataset/labels/train'
    train_model(img_dir, label_dir)
