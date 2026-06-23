import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model_manager import get_model
from training_config import TRAINING_CONFIG



def start_webcam(style_name,device,alpha=1.0):
    config = TRAINING_CONFIG[style_name]
    MODEL_PATH = config["save_path"]
    DEVICE = device

    model = get_model(MODEL_PATH,DEVICE)

    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])

    cap = cv2.VideoCapture(0)           # opens camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )
        )

        image = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(image)

        output = alpha*output + (1-alpha)*image
        output = output.squeeze(0)
        output = output.clamp(0,1)
        output = transforms.ToPILImage()(output.cpu())

        output_frame = cv2.cvtColor(
            np.array(output),
            cv2.COLOR_RGB2BGR
        )

        cv2.imshow(
            "Stylized Webcam",
            output_frame
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()