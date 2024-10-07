import torch
import torch.ao.quantization
from torchvision import models
import os
# Carica il modello pre-addestrato
model = models.vgg16(pretrained=True)
model.eval()

# Applica la quantizzazione dinamica
quantized_model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# Salva il modello quantizzato
os.makedirs("minimizedModels/quantization", exist_ok=True)
torch.save(quantized_model.state_dict(), "minimizedModels/quantization/vgg16_dynamic_quantized.pth")
