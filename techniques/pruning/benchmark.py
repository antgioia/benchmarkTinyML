import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
import time
import os

# Imposta il dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory base per i modelli pruned
base_dir = "minimizedModels/pruning"

# Configurazione per ogni modello e relativa trasformazione
model_configs = {
    "alexnet": {
        "model_class": models.alexnet,
        "transform": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        "data_loader": None,  # Verrà popolato dinamicamente
        "num_classes": None  # Verrà popolato dinamicamente
    },
    "resnet18": {
        "model_class": models.resnet18,
        "transform": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        "data_loader": None,  # Verrà popolato dinamicamente
        "num_classes": None  # Verrà popolato dinamicamente
    },
    "vgg16": {
        "model_class": models.vgg16,
        "transform": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        "data_loader": None,  # Verrà popolato dinamicamente
        "num_classes": None  # Verrà popolato dinamicamente
    }
}

# Caricamento del dataset per ogni modello
def setup_data_loaders():
    for model_name, config in model_configs.items():
        if model_name == "alexnet" or model_name == "vgg16":
            batch_size = 128
        elif model_name == "resnet18":
            batch_size = 256
        dataset = datasets.ImageFolder(root='dataset/imagenet-mini/train', transform=config["transform"])
        config["data_loader"] = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        config["num_classes"] = len(dataset.classes)

# Funzione per salvare i risultati in un file di testo
def save_results(output_dir, model_name, pruning_technique, accuracy, precision, recall, auc_roc, inference_time, throughput, model_size):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{model_name}_{pruning_technique}_results.txt")
    
    with open(file_path, 'w') as f:
        f.write(f"Modello: {model_name}\n")
        f.write(f"Tecnica di Pruning: {pruning_technique}\n")
        f.write(f"Accuratezza: {accuracy:.4f}\n")
        f.write(f"Precisione: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"AUC-ROC: {auc_roc:.4f}\n")
        f.write(f"Tempo di inferenza: {inference_time:.4f} secondi\n")
        f.write(f"Throughput: {throughput:.4f} immagini/secondo\n")
        f.write(f"Dimensione del modello: {model_size:.4f} MB\n")
        f.write("-" * 50 + "\n")

# Funzione per caricare i modelli minimizzati
def load_model(model_path, model_type, num_classes):
    model_class = model_configs[model_type]["model_class"]
    model = model_class(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Funzione per valutare le prestazioni del modello
def evaluate_model(model, dataloader, device):
    model.to(device)
    model.eval()
    
    start_time = time.time()

    all_preds = []
    all_probs = []
    all_labels = []
    
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

# Funzione principale per eseguire il benchmark
if __name__ == "__main__":
    setup_data_loaders()

    # Definisci le tecniche di pruning da eseguire per ciascun modello
    pruning_techniques = ["pruned_casuale_non_strutturato", "pruned_global", "pruned_non_strutturato", "pruned_strutturato_per_canali"]

    for model_name, config in model_configs.items():
        for pruning_technique in pruning_techniques:
            model_path = os.path.join(base_dir, f"{model_name}_{pruning_technique}.pth")
            
            if os.path.exists(model_path):
                print(f"Benchmarking {model_name} con tecnica {pruning_technique}...")
                
                # Carica il modello pruned
                model = load_model(model_path, model_name, config["num_classes"])

                # Valuta il modello
                accuracy, precision, recall, auc_roc, inference_time, throughput = evaluate_model(model, config["data_loader"], device)

                # Calcola la dimensione del modello
                model_size = os.path.getsize(model_path) / (1024 ** 2)

                # Salva i risultati in un file di testo
                save_results("benchmark/pruning/"+ model_name, model_name, pruning_technique, accuracy, precision, recall, auc_roc, inference_time, throughput, model_size)

            else:
                print(model_path)
                print(f"Il modello {model_name} ({pruning_technique}) non esiste. Controlla il percorso.")
