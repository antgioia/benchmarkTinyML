import sys
import torch
from torchvision import transforms, models
import os
from codecarbon import EmissionsTracker

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
        "data_loader": None,
        "num_classes": 1000  
    },
    "resnet18": {
        "model_class": models.resnet18,
        "transform": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        "data_loader": None, 
        "num_classes": 1000 
    },
    "vgg16": {
        "model_class": models.vgg16,
        "transform": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        "data_loader": None,
        "num_classes": 1000
    }
}

# Caricamento del dataset per ogni modello
def setup_data_loaders(loader_path_Alexnet, loader_path_Resnet18, loader_path_Vgg16):
    for model_name, config in model_configs.items():
        if model_name == "alexnet":
            config["data_loader"] = torch.load(loader_path_Alexnet)
        elif model_name == "resnet18":
            config["data_loader"] = torch.load(loader_path_Resnet18)
        elif model_name == "vgg16":
            config["data_loader"] = torch.load(loader_path_Vgg16)

# Funzione per caricare i modelli minimizzati
def load_model(model_path, model_type, num_classes):
    model_class = model_configs[model_type]["model_class"]
    model = model_class(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Funzione per valutare il modello
def evaluate_model(model, dataset_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataset_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# Funzione per eseguire il benchmark energetico
def benchmark_energy(model, model_name, technique, dataset_loader, counter):
    # Inizializza CodeCarbon
    output_dir = f"energyConsumed/pruning/{model_name}/{technique}"
    os.makedirs(output_dir, exist_ok=True)
    tracker = EmissionsTracker(output_dir=output_dir, output_file=f"{technique}_{model_name}_{counter}.csv")
    
    # Inizia il monitoraggio energetico
    tracker.start()
    
    # Esegui la valutazione
    accuracy = evaluate_model(model, dataset_loader)
    print("ACCURACY = ", accuracy)

    # Termina il monitoraggio energetico
    tracker.stop()

# Funzione principale per eseguire il benchmark
if __name__ == "__main__":
    loader_path_Alexnet = sys.argv[1]
    loader_path_Resnet18 = sys.argv[2]
    loader_path_Vgg16 = sys.argv[3]

    setup_data_loaders(loader_path_Alexnet, loader_path_Resnet18, loader_path_Vgg16)

    # Definisci le tecniche di pruning da eseguire per ciascun modello
    pruning_techniques = ["pruned_casuale_non_strutturato", "pruned_global", "pruned_non_strutturato", "pruned_strutturato_per_canali"]

    for model_name, config in model_configs.items():
        for pruning_technique in pruning_techniques:
            model_path = os.path.join(base_dir, f"{model_name}_{pruning_technique}.pth")
            
            if os.path.exists(model_path):
                print(f"Benchmarking {model_name} con tecnica {pruning_technique}...")
                
                # Carica il modello pruned
                model = load_model(model_path, model_name, config["num_classes"])
                counter = 0
                for counter in range(100):
                    benchmark_energy(model, model_name, pruning_technique, config["data_loader"], counter)
            else:
                print(model_path)
                print(f"Il modello {model_name} ({pruning_technique}) non esiste. Controlla il percorso.")
