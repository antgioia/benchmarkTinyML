import torch
import torch.ao.quantization
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

# Imposta il dispositivo (GPU o CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica il modello pre-addestrato
model = models.resnet18(pretrained=True)
model.eval()
model.to(device)

# Applica la quantizzazione dinamica
quantized_model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# Prepara il dataset di validazione
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(root='dataset/imagenet-mini/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Funzione per calcolare l'accuratezza
def calculate_accuracy(model, data_loader):
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

# Calcola l'accuratezza del modello quantizzato
accuracy = calculate_accuracy(quantized_model, val_loader)
print(f'Accuratezza del modello quantizzato: {accuracy:.4f}')

torch.save(quantized_model, "minimizedModels/quantization/resnet18_dynamic_quantized.pth")
