import torch

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b,c,h*w)
    gram = torch.bmm(features, features.transpose(1,2))
    return gram/(h*w)