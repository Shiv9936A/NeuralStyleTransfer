import torch
import torch.nn as nn
import torch.optim as optim                     # optimizers
from tqdm import tqdm
from torchvision import transforms              # image preprocessing
from torch.utils.data import DataLoader         # creates batches
from PIL import Image
from dataset import FlatImageDataset
from models.transform_net import TransformNet
from models.vgg import VGGFeatures
from utils import gram_matrix
from training_config import TRAINING_CONFIG
import json
import time
import random
import os

torch.backends.cudnn.benchmark = True

IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 50

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()         # checks GPU availability
    else "cpu"
)

DATA_DIR = "data"
# STYLE_IMAGE = "styles/hexagons.png"
# MODEL_SAVE_PATH = "saved_models/style_model.pth"

def train_style(style_name):
    config = TRAINING_CONFIG[style_name]
    MODEL_SAVE_PATH = config["save_path"]
    style_folder = config["style_folder"]
    style_files = [f for f in os.listdir(style_folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not style_files:
        raise FileNotFoundError(f"No images found in {style_folder}")
    style_path = os.path.join(style_folder, style_files[0])
    print(f"Using style image: {style_path}")

    style_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    content_transform = transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE*1.2)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor()                           # jpg/jpeg/png to Tensor [3,256,256]
    ])

    dataset = FlatImageDataset(DATA_DIR,content_transform)
    total_images = len(dataset)
    print(f"Dataset Size: {total_images}")

    loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2,pin_memory=True)

    model = TransformNet().to(DEVICE)
    vgg = VGGFeatures().to(DEVICE)
    vgg.eval()
    mse = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr = 5e-4
    )


    for param in vgg.parameters():              # frozen vgg training
        param.requires_grad = False
    vgg.eval()

    best_loss = float("inf")
    loss_history = []
    start_time = time.time()

    style_img = Image.open(style_path).convert("RGB")
    style_img = style_transform(style_img).unsqueeze(0).to(DEVICE)
    style_features = vgg(style_img)

    style_gram = []
    for f in style_features:
        # f shape: [1, channels, h, w]
        gram = gram_matrix(f)  # [1, c, c]
        gram = gram.squeeze(0).detach()  # [c, c]
        style_gram.append(gram)
        
    print("Training started")
    for epoch in range(EPOCHS):
        # style_file = random.choice(os.listdir(style_folder))
        # style_path = os.path.join( style_folder,style_file)
        print(f"Using Style Image: {style_path}")
        total_epoch_loss = 0

        progress_bar = tqdm(
            loader,
            desc=f"Epoch {epoch+1}/{EPOCHS}"
        )

        for content in progress_bar:
            content = content.to(DEVICE)
            optimizer.zero_grad()
            output = model(content)
            content_features = vgg(content)
            output_features = vgg(output)

            content_loss = mse(
                output_features[0],
                content_features[0]
            )

            style_loss = 0
            style_weights = [1.0,1.0,1.0,1.0]
            for of, sg, w in zip(output_features, style_gram, style_weights):
                gm = gram_matrix(of)
                gm_avg = gm.mean(0)
                style_loss += w*mse(
                    gm_avg,
                    sg
                )

            loss = content_loss*1.0 + style_loss * 1e4
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # with autocast(device_type="cuda"):

            #     output = model(content)

            #     content_features = vgg(content)

            #     output_features = vgg(output)

            #     content_loss = mse(
            #         output_features[2],
            #         content_features[2]
            #     )

            #     style_loss = 0

            #     for of, sg in zip(output_features, style_gram):

            #         gm = gram_matrix(of)

            #         style_loss += mse(
            #             gm,
            #             sg.expand_as(gm)
            #         )

            #     loss = content_loss + style_loss * 100


            # scaler.scale(loss).backward()

            # scaler.step(optimizer)

            # scaler.update()

            current_loss = loss.item()
            loss_history.append(current_loss)
            total_epoch_loss += current_loss
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_epoch_loss / len(loader)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH.replace(".pth","_best.pth"))
            print(f"  ✓ Saved best model with loss: {best_loss:.4f}")


    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    metadata = {
        "style": style_name,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "best_loss": best_loss
    }
    with open(
        f"saved_models/{style_name}_metadata.json",
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )
    with open(f"loss_history_{style_name}.json","w") as f:
        json.dump(loss_history,f)


    end_time = time.time()

    stats = {
        "style": style_name,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "dataset_size": total_images,
        "best_loss": best_loss,
        "training_time_sec": round(
            end_time - start_time,
            2
        )
    }

    with open(
        f"saved_models/{style_name}_stats.json",
        "w"
    ) as f:
        json.dump(
            stats,
            f,
            indent=4
        )

    print("Model successfully saved")


if __name__ == "__main__":
    import sys

    style = sys.argv[1] if len(sys.argv) > 1 else "anime"
    train_style(style)
    
