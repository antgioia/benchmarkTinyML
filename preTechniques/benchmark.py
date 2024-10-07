import torch
import torchvision.models as models
import torch.nn.functional as F
import os
import time
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, alexnet, vgg16, ResNet18_Weights, VGG16_Weights, AlexNet_Weights

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

# Funzione principale per eseguire il benchmark
def run_benchmark(model, model_name, loader):
    # Calcola le metriche di prestazione
    accuracy, precision, recall, auc_roc, inference_time, throughput = evaluate_model(model, loader, device)

    # Calcola la dimensione del modello
    model_size = sum(param.numel() for param in model.parameters()) * 4 / (1024 * 1024)  # in MB

    # Salva i risultati
    save_results("benchmark/prima/" + model_name, model_name, accuracy, precision, recall, auc_roc, inference_time, throughput, model_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alexnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

resnet18_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

vgg16_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Caricamento del dataset
datasetAlexnet = ImageFolder("dataset/imagenet-mini/train", transform=alexnet_transform)
loaderAlexNet = DataLoader(datasetAlexnet, batch_size=128, shuffle=False)
datasetResnet18 = ImageFolder("dataset/imagenet-mini/train", transform=resnet18_transform)
loaderResNet18 = DataLoader(datasetResnet18, batch_size=256, shuffle=False)
datasetVgg16 = ImageFolder("dataset/imagenet-mini/train", transform=vgg16_transform)
loaderVgg16 = DataLoader(datasetVgg16, batch_size=128, shuffle=False)

weightsAlexNet = AlexNet_Weights.IMAGENET1K_V1
modelAlexNet = alexnet(weights=weightsAlexNet)
modelAlexNet.eval()

weightsResNet18 = ResNet18_Weights.IMAGENET1K_V1
modelResNet18 = resnet18(weights=weightsResNet18)
modelResNet18.eval()

weightsVgg16 = VGG16_Weights.IMAGENET1K_V1
modelVgg16 = vgg16(weights=weightsVgg16)
modelVgg16.eval()

models_to_test = {
    "alexnet": (modelAlexNet, loaderAlexNet),
    "resnet18": (modelResNet18, loaderResNet18),
    "vgg16": (modelVgg16, loaderVgg16)
 }

# Esecuzione del benchmark per ogni modello
for model_name, (model, loader) in models_to_test.items():
    print("Valutazione: ", model_name)
    run_benchmark(model, model_name, loaderResNet18)