from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import os

class FlatImageDataset(Dataset):
    def __init__(self, image_dir, transform = None):
       
        self.image_paths = []
        
        for file in os.listdir(image_dir):

            extension = os.path.splitext(file)[1].lower()

            if extension in [".jpg", ".jpeg", ".png"]:

                full_path = os.path.join(
                    image_dir,
                    file
                )

                self.image_paths.append(
                    full_path
                )

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, key):
        image = default_loader(self.image_paths[key])
        image = image.convert("RGB")    # force RGB — handles RGBA PNGs
        if self.transform:
            image = self.transform(image)

        return image