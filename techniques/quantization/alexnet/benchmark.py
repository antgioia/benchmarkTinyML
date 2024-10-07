import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import time

# Imposta il dispositivo (GPU o CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.alexnet(pretrained=True)
model.eval()

# Applica la quantizzazione dinamica
quantized_model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# Salva il modello quantizzato
os.makedirs("minimizedModels/quantization", exist_ok=True)
torch.save(quantized_model.state_dict(), "minimizedModels/quantization/alexnet_dynamic_quantized.pth")

# Carica i pesi quantizzati salvati
quantized_model.load_state_dict(torch.load("minimizedModels/quantization/alexnet_dynamic_quantized.pth"))
quantized_model.to(device)

# Prepara le trasformazioni per le immagini di input
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_dataset_alexnet = datasets.ImageFolder(root='dataset/imagenet-mini/train', transform=transform)
val_loader_alexnet = DataLoader(dataset=val_dataset_alexnet, batch_size=128, shuffle=False)

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

accuracy, precision, recall, auc_roc, inference_time, throughput =  evaluate_model(quantized_model, val_loader_alexnet, device)

# Calcola la dimensione del modello
model_size = os.path.getsize("minimizedModels/quantization/alexnet_dynamic_quantized.pth") / (1024 ** 2)

# Funzione per salvare i risultati in un file di testo
def save_results(output_dir, model_name, accuracy, precision, recall, auc_roc, inference_time, throughput, model_size):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{model_name}_quantization_dynamic_results.txt")
    print("PATH", file_path)
    
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

# Salva i risultati in un file di testo
save_results("benchmark/quantization/alexnet/", "alexnet", accuracy, precision, recall, auc_roc, inference_time, throughput, model_size)
