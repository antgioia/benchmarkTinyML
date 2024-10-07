import os
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
import torchvision.models as models
import torch.nn.functional as F

# Imposta il dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica il modello salvato
class LowRankResNet18(torch.nn.Module):
    def __init__(self):
        super(LowRankResNet18, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Applica low-rank approximation ai layer convoluzionali (da fare in precedenza)

    def forward(self, x):
        return self.model(x)

def evaluate_model(model, dataloader, device):
    model.to(device)
    model.eval()
    
    start_time = time.time()

    all_preds = []
    all_probs = []
    all_labels = []
    
    total_time = 0  # Inizializza il tempo totale
    total_images = 0  # Inizializza il contatore delle immagini
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            total_images += images.size(0)  # Aggiorna il contatore delle immagini
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calcola le metriche
    inference_time = time.time() - start_time
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    auc_roc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    
    # Calcola il tempo medio di inferenza e il throughput
    throughput = total_images / inference_time if inference_time > 0 else 0
    
    return accuracy, precision, recall, auc_roc, inference_time, throughput

# Funzione per salvare i risultati in un file di testo
def save_results(output_dir, model_name, accuracy, precision, recall, auc_roc, inference_time, throughput, model_size):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{model_name}_results.txt")
    
    with open(file_path, 'w') as f:
        f.write(f"Modello: {model_name}\n")
        f.write(f"Accuratezza: {accuracy:.4f}\n")
        f.write(f"Precisione: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"AUC-ROC: {auc_roc:.4f}\n")
        f.write(f"Tempo di inferenza: {inference_time:.4f} secondi\n")
        f.write(f"Throughput: {throughput:.4f} immagini/secondo\n")
        f.write(f"Dimensione del modello: {model_size:.4f} MB\n")
        f.write("-" * 50 + "\n")

# Carica il modello ottimizzato
model = LowRankResNet18()
model.load_state_dict(torch.load('minimizedModels/lowRankApproximation/resnet18_low_rank.pth'), strict=False)
model.eval()  # Imposta il modello in modalità valutazione

# Trasformazioni per il preprocessing del dataset
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Carica il dataset ImageNet-mini
imagenet_val_dir = 'dataset/imagenet-mini/train'  # Modifica con il tuo percorso
val_dataset = datasets.ImageFolder(imagenet_val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# Calcola le metriche
accuracy, precision, recall, auc_roc, inference_time, throughput = evaluate_model(model, val_loader, device)

# Calcola la dimensione del modello
model_size = os.path.getsize("minimizedModels/lowRankApproximation/resnet18_low_rank.pth") / (1024 ** 2)

# Salva i risultati in un file di testo
save_results("benchmark/lowRankApproximation/resnet18", "resnet18", accuracy, precision, recall, auc_roc, inference_time, throughput, model_size)
