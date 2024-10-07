import sys
import torch
import torchvision.models as models
import torch.nn.functional as F
import os
from codecarbon import EmissionsTracker

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
    output_dir = f"energyConsumed/pre/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    tracker = EmissionsTracker(output_dir=output_dir, output_file=f"{technique}_{model_name}_{counter}.csv")
    
    # Inizia il monitoraggio energetico
    tracker.start()
    
    # Esegui la valutazione
    accuracy = evaluate_model(model, dataset_loader)
    print("ACCURACY = ", accuracy)

    # Termina il monitoraggio energetico
    tracker.stop()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loaderAlexNet = torch.load(sys.argv[1])
loaderResNet18 = torch.load(sys.argv[2])
loaderVgg16 = torch.load(sys.argv[3])

# Caricamento dei modelli pre-addestrati
models_to_test = {
    "alexnet": (models.alexnet(pretrained=True).to(device), loaderAlexNet),
    "resnet18": (models.resnet18(pretrained=True).to(device), loaderResNet18),
    "vgg16": (models.vgg16(pretrained=True).to(device), loaderVgg16)
}

# Esecuzione del benchmark per ogni modello
for model_name, (model, loader) in models_to_test.items():
    print("Valutazione: ", model_name)
    counter = 0
    for counter in range(100):
        benchmark_energy(model, model_name, "pre", loader, counter)