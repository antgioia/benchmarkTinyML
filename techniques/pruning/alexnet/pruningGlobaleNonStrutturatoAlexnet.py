import torch
import torch.nn.utils.prune as prune
import os
import torchvision.models as models


# Carica il modello alexnet pre-addestrato
model = models.alexnet(pretrained=True)

# Definisce i parametri da prunare (tutti gli strati convoluzionali)
parameters_to_prune = []
for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        parameters_to_prune.append((module, 'weight'))

# Applica pruning globale non strutturato
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3
)

# Rimuovi le maschere di pruning e mantieni solo i pesi prunati
for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.remove(module, 'weight')

# Salva il modello pruned
os.makedirs("minimizedModels/pruning", exist_ok=True)
torch.save(model.state_dict(), os.path.join("minimizedModels/pruning", 'alexnet_pruned_global.pth'))
