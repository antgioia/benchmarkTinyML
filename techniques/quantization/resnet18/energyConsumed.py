import sys
import torch
import os

# Imposta il dispositivo (GPU o CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica il modello quantizzato
quantized_model = torch.load("minimizedModels/quantization/resnet18_dynamic_quantized.pth")
quantized_model.eval()
quantized_model.to(device)

loaderResnet = torch.load(sys.argv[1])

import torch
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
    output_dir = f"energyConsumed/{technique}/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    tracker = EmissionsTracker(output_dir=output_dir, output_file=f"quanitzation_dynamic_{model_name}_{counter}.csv")
    
    # Inizia il monitoraggio energetico
    tracker.start()
    
    # Esegui la valutazione
    accuracy = evaluate_model(model, dataset_loader)
    print("ACCURACY = ", accuracy)

    # Termina il monitoraggio energetico
    tracker.stop()

counter = 0
for counter in range(100):
    benchmark_energy(quantized_model, "resnet18", "quantization", loaderResnet, counter)