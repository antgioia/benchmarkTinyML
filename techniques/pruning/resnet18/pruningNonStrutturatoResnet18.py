import torch
import torch.nn.utils.prune as prune
import torchvision.models as models
import os

# Carica il modello ResNet18 pre-addestrato
model = models.resnet18(pretrained=True)

# Applica pruning non strutturato L1 su tutti gli strati Conv2d
for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.3)

# Rimuovi le maschere di pruning e mantieni solo i pesi prunati
for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.remove(module, 'weight')

# Salva il modello pruned
os.makedirs("minimizedModels/pruning", exist_ok=True)
torch.save(model.state_dict(), os.path.join("minimizedModels/pruning", 'resnet18_pruned_non_strutturato.pth'))
