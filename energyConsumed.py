import os
import subprocess
import torch
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import random

dataset_path = "dataset/imagenet-mini/val"

# Carica il dataset originale
full_dataset = ImageFolder(dataset_path)

# Seleziona 1000 immagini casuali una sola volta
indices = random.sample(range(len(full_dataset)), 1000)
subset = Subset(full_dataset, indices)

alexnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

resnet18_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

vgg16_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Crea dataset separati con le trasformazioni per ciascun modello
dataset_alexnet = Subset(ImageFolder(dataset_path, transform=alexnet_transform), indices)
dataset_resnet = Subset(ImageFolder(dataset_path, transform=resnet18_transform), indices)
dataset_vgg = Subset(ImageFolder(dataset_path, transform=vgg16_transform), indices)

# Crea i DataLoader per ciascun modello
loader_alexnet = DataLoader(dataset_alexnet, batch_size=128, shuffle=False)
loader_resnet = DataLoader(dataset_resnet, batch_size=256, shuffle=False)
loader_vgg = DataLoader(dataset_vgg, batch_size=128, shuffle=False)

os.makedirs("energyConsumedUtils", exist_ok=True)
loader_alexnet_path = "energyConsumedUtils/alexnet_loader.pth"
loader_resnet_path = "energyConsumedUtils/resnet18_loader.pth"
loader_vgg16_path = "energyConsumedUtils/vgg16_loader.pth"
torch.save(loader_alexnet, loader_alexnet_path)
torch.save(loader_resnet, loader_resnet_path)
torch.save(loader_vgg, loader_vgg16_path)

try:
    subprocess.run(['python', 'techniques/quantization/energyConsumed.py', loader_alexnet_path, loader_resnet_path, loader_vgg16_path], check=True)
    subprocess.run(['python', 'techniques/pruning/energyConsumed.py', loader_alexnet_path, loader_resnet_path, loader_vgg16_path], check=True)
    subprocess.run(['python', 'techniques/lowRankApproximation/energyConsumed.py', loader_alexnet_path, loader_resnet_path, loader_vgg16_path], check=True)
    subprocess.run(['python', 'preTechniques/energyConsumed.py', loader_alexnet_path, loader_resnet_path, loader_vgg16_path], check=True)
except subprocess.CalledProcessError as e:
    print(f"Errore nell'eseguire: {e}\n")