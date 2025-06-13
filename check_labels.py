import os

label_dir = '/home/francesca/Scrivania/ResNetBBoxRegressor/Dataset/labels/train'

# Lista di tutti i file .txt (ordinati)
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

# Stampiamo le prime 3 label
for i, filename in enumerate(label_files[:3]):
    filepath = os.path.join(label_dir, filename)
    with open(filepath, 'r') as f:
        line = f.readline().strip()
        values = line.split()
    print(f"File: {filename}")
    print(f"Values: {values}")
    print(f"Numero valori: {len(values)}\n")
