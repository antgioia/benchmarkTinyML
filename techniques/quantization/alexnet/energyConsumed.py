import sys
import torch
from torchvision import models
from codecarbon import EmissionsTracker
import os

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
    output_dir = f"energyConsumed/{technique}/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    tracker = EmissionsTracker(output_dir=output_dir, output_file=f"quantization_dynamic_{model_name}_{counter}.csv")
    
    # Inizia il monitoraggio energetico
    tracker.start()
    
    # Esegui la valutazione
    accuracy = evaluate_model(model, dataset_loader)
    print("ACCURACY = ", accuracy)

    # Termina il monitoraggio energetico
    tracker.stop()

# Imposta il dispositivo (GPU o CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.alexnet(pretrained=True)
model.eval()

# Applica la quantizzazione dinamica
quantized_model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
quantized_model.to(device)

loaderAlexnet = torch.load(sys.argv[1])

counter = 0
for counter in range(100):
    benchmark_energy(quantized_model, "alexnet", "quantization", loaderAlexnet, counter)