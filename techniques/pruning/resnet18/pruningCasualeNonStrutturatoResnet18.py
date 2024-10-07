import torch
import torch.nn.utils.prune as prune
import os
import torchvision.models as models

# Carica il modello resnet18 pre-addestrato
model = models.resnet18(pretrained=True)

# Applica pruning casuale non strutturato su tutti gli strati Conv2d
for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.random_unstructured(module, name='weight', amount=0.01)
    
# Rimuovi le maschere di pruning e mantieni solo i pesi prunati
for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.remove(module, 'weight')

# Salva il modello pruned
os.makedirs("minimizedModels/pruning", exist_ok=True)
torch.save(model.state_dict(), os.path.join("minimizedModels/pruning", 'resnet18_pruned_casuale_non_strutturato.pth'))
