# 1. DEBUG DEL DATASET - Controlla la distribuzione dei dati
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch

def analyze_dataset_distribution(dataset):
    """Analizza la distribuzione delle bounding box nel dataset"""
    centers_x = []
    centers_y = []
    widths = []
    heights = []
    
    for i in range(len(dataset)):
        _, bbox = dataset[i]
        x, y, w, h = bbox
        centers_x.append(x)
        centers_y.append(y)
        widths.append(w)
        heights.append(h)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0,0].hist(centers_x, bins=50, alpha=0.7)
    axes[0,0].set_title('Distribuzione Center X')
    axes[0,0].set_xlabel('Center X')
    
    axes[0,1].hist(centers_y, bins=50, alpha=0.7)
    axes[0,1].set_title('Distribuzione Center Y')
    axes[0,1].set_xlabel('Center Y')
    
    axes[1,0].hist(widths, bins=50, alpha=0.7)
    axes[1,0].set_title('Distribuzione Width')
    axes[1,0].set_xlabel('Width')
    
    axes[1,1].hist(heights, bins=50, alpha=0.7)
    axes[1,1].set_title('Distribuzione Height')
    axes[1,1].set_xlabel('Height')
    
    plt.tight_layout()
    plt.savefig('dataset_distribution.png')
    plt.show()
    
    print(f"Center X: min={min(centers_x):.2f}, max={max(centers_x):.2f}, mean={np.mean(centers_x):.2f}")
    print(f"Center Y: min={min(centers_y):.2f}, max={max(centers_y):.2f}, mean={np.mean(centers_y):.2f}")
    print(f"Width: min={min(widths):.2f}, max={max(widths):.2f}, mean={np.mean(widths):.2f}")
    print(f"Height: min={min(heights):.2f}, max={max(heights):.2f}, mean={np.mean(heights):.2f}")