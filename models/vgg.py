from torchvision import models
import torch.nn as nn

class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.features = vgg.features[:21]       # first 21 layers - styling ; deep layers - classification
        for param in self.features.parameters():
            param.requires_grad = False         # no training of VGG

    def forward(self,x):
        outputs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in [0,5,10]:       # 0 layer - edges ; 5 - textures ; 10 - patterns ; 19 - high level content
                outputs.append(x)
            
        return outputs    