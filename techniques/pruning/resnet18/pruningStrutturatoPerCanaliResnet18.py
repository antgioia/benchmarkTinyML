import torch
import torch.nn.utils.prune as prune
import torchvision.models as models
import os

# Carica il modello ResNet18 pre-addestrato
model = models.resnet18(pretrained=True)

# Applica pruning strutturato LN per canali su tutti gli strati Conv2d
for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.ln_structured(module, name='weight', amount=0.01, n=2, dim=0)

# Rimuovi le maschere di pruning e mantieni solo i pesi prunati
for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.remove(module, 'weight')

# Salva il modello pruned
os.makedirs("minimizedModels/pruning", exist_ok=True)
torch.save(model.state_dict(), os.path.join("minimizedModels/pruning", 'resnet18_pruned_strutturato_per_canali.pth'))
