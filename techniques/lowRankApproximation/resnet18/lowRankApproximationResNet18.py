import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# Carica il modello ResNet18 pre-addestrato
resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Esempio di decomposizione su un singolo layer convoluzionale
def low_rank_approximation(conv_layer, rank):
    """
    Funzione che approssima una convoluzione esistente usando una decomposizione a basso rango
    :param conv_layer: Layer convoluzionale da decomporre
    :param rank: Il rango per l'approssimazione
    """
    # Estrai i pesi della convoluzione
    weight = conv_layer.weight.data
    out_channels, in_channels, h, w = weight.shape
    
    # Applica la decomposizione SVD sulla matrice di peso
    weight_reshaped = weight.view(out_channels, -1)
    U, S, V = torch.svd(weight_reshaped)
    
    # Truncate i componenti della SVD in base al rango desiderato
    U_truncated = U[:, :rank]
    S_truncated = torch.diag(S[:rank])
    V_truncated = V[:, :rank]
    
    # Ricostruisci i pesi approssimati
    low_rank_weights = U_truncated @ S_truncated @ V_truncated.t()
    
    # Ricostruisci la nuova convoluzione usando il nuovo peso
    low_rank_weights = low_rank_weights.view(out_channels, in_channels, h, w)
    new_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(h, w), stride=conv_layer.stride, padding=conv_layer.padding)
    new_conv.weight.data = low_rank_weights
    
    return new_conv

# Applica l'approssimazione a basso rango al primo layer convoluzionale
resnet18.conv1 = low_rank_approximation(resnet18.conv1, rank=10)

# Salva il modello ottimizzato
os.makedirs("minimizedModels/lowRankApproximation", exist_ok=True)
torch.save(resnet18.state_dict(), 'minimizedModels/lowRankApproximation/resnet18_low_rank.pth')
