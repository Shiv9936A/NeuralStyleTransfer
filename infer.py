import torch
from torchvision import transforms
from PIL import Image
from models.transform_net import TransformNet
from model_manager import get_model

# MODEL_PATH = "saved_models/style_model.pth"
# INPUT_IMAGE = "test/input.jpg"
# OUTPUT_IMAGE = "outputs/output.jpg"
# IMAGE_SIZE = 256
# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available()
#     else "cpu"
# )

def stylize_image(image, model_path,device,alpha=1.0):
    alpha = max(0.0,min(alpha,1.0))             # a just validation
    MODEL_PATH = model_path
    DEVICE = device
    IMAGE_SIZE = 512

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE),interpolation=Image.LANCZOS),
        transforms.ToTensor(),
    ])

    model = get_model(MODEL_PATH,DEVICE)
    print("="*50)
    print("LOADED MODEL:", MODEL_PATH)
    print("="*50)                         

    # image = Image.open(INPUT_IMAGE).convert("RGB")
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)


    with torch.no_grad():                       # because of no gradient , faster
        output = model(image)
        output = torch.clamp(output, 0, 1)
        # output = output * 1.25                # brightness boost 

    output = alpha*output + (1-alpha)*image
    output = output.squeeze(0)
    output = output.clamp(0,1)                  # output will be adjusted between 0 - 1
    output_image = transforms.ToPILImage()(output.cpu())
    # output_image.save(OUTPUT_IMAGE)
    # print(f"Saved: {OUTPUT_IMAGE}")

    return output_image



