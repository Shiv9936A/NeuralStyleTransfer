from torchvision import models
import torch.nn as nn
import torch

class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.features = vgg.features[:21]       # first 21 layers - styling ; deep layers - classification
        for param in self.features.parameters():
            param.requires_grad = False         # no training of VGG
            
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))


    def forward(self,x):
        x = (x-self.mean)/self.std
        outputs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in [0,5,10,19]:       # 0 layer - edges ; 5 - textures ; 10 - patterns ; 19 - high level content
                outputs.append(x)
            
        return outputs    