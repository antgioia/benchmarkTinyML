import sys
import torch
from torchvision import models
import os
from torchvision.models import VGG16_Weights, ResNet18_Weights, AlexNet_Weights
from codecarbon import EmissionsTracker

# Imposta il dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory base per i modelli pruned
base_dir = "minimizedModels/lowRankApproximation"

# Configurazione per ogni modello e relativa trasformazione
model_configs = {
    "alexnet": {
        "model_class": models.alexnet,
        "data_loader": None,
        "num_classes": 1000  
    },
    "resnet18": {
        "model_class": models.resnet18,
        "data_loader": None, 
        "num_classes": 1000 
    },
    "vgg16": {
        "model_class": models.vgg16,
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
def benchmark_energy(model, model_name, dataset_loader, counter):
    # Inizializza CodeCarbon
    output_dir = f"energyConsumed/lowRankApproximation/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    tracker = EmissionsTracker(output_dir=output_dir, output_file=f"low_rank_{model_name}_{counter}.csv")
    
    # Inizia il monitoraggio energetico
    tracker.start()
    
    # Esegui la valutazione
    accuracy = evaluate_model(model, dataset_loader)
    print("ACCURACY = ", accuracy)

    # Termina il monitoraggio energetico
    tracker.stop()

class LowRankResNet18(torch.nn.Module):
    def __init__(self):
        super(LowRankResNet18, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    def forward(self, x):
        return self.model(x)
    
class LowRankAlexNet(torch.nn.Module):
    def __init__(self):
        super(LowRankAlexNet, self).__init__()
        self.model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

    def forward(self, x):
        return self.model(x)
    
class LowRankVGG16(torch.nn.Module):
    def __init__(self):
        super(LowRankVGG16, self).__init__()
        self.model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

    def forward(self, x):
        return self.model(x)

# Funzione principale per eseguire il benchmark
if __name__ == "__main__":
    loader_path_Alexnet = sys.argv[1]
    loader_path_Resnet18 = sys.argv[2]
    loader_path_Vgg16 = sys.argv[3]

    setup_data_loaders(loader_path_Alexnet, loader_path_Resnet18, loader_path_Vgg16)

    for model_name, config in model_configs.items():
        model_path = os.path.join(base_dir, f"{model_name}_low_rank.pth")
        
        if os.path.exists(model_path):
            print(f"Benchmarking {model_name}...")
            if model_name == "resnet18":
                model = LowRankResNet18()
            elif model_name == "vgg16":
                model = LowRankVGG16()
            elif model_name == "alexnet":
                model = LowRankAlexNet()
            model.load_state_dict(torch.load(model_path), strict=False)
            model.eval()

            counter = 0
            for counter in range(100):
                benchmark_energy(model, model_name, config["data_loader"], counter)
        else:
            print(model_path)
            print(f"Il modello {model_name} non esiste. Controlla il percorso.")
