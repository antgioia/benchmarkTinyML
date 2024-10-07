import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights

# Carica il modello VGG16 pre-addestrato
vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

# Funzione per decomposizione a basso rango di un layer convoluzionale
def low_rank_approximation(conv_layer, rank):
    """
    Funzione che approssima una convoluzione esistente usando una decomposizione a basso rango
    :param conv_layer: Layer convoluzionale da decomporre
    :param rank: Il rango per l'approssimazione
    """
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
    low_rank_weights = low_rank_weights.view(out_channels, in_channels, h, w)
    
    # Crea un nuovo layer convoluzionale con i pesi approssimati
    new_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(h, w), stride=conv_layer.stride, padding=conv_layer.padding)
    new_conv.weight.data = low_rank_weights
    
    return new_conv

# Applica l'approssimazione a basso rango al primo layer convoluzionale di VGG16
vgg16.features[0] = low_rank_approximation(vgg16.features[0], rank=10)

# Salva il modello ottimizzato
os.makedirs("minimizedModels/lowRankApproximation", exist_ok=True)
torch.save(vgg16.state_dict(), 'minimizedModels/lowRankApproximation/vgg16_low_rank.pth')
