import cv2
import torch
import time
import numpy as np
from PIL import Image               # why PIL? because torchvision transform works better with them
from torchvision import transforms
from models.transform_net import TransformNet
from model_manager import get_model

# MODEL_PATH = "saved_models/style_model.pth"
# INPUT_VIDEO = "test/input.mp4"
# OUTPUT_VIDEO = "outputs/output.mp4"
# IMAGE_SIZE = 256
# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available()
#     else "cpu"
# )

def stylize_video(input_video,output_video,model_path,device):
    MODEL_PATH = model_path
    INPUT_VIDEO = input_video
    OUTPUT_VIDEO = output_video
    IMAGE_SIZE = 256
    DEVICE = device

    model = get_model(MODEL_PATH,DEVICE)


    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
    ])


    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),        # fourcc = 4-charachter code {"mp4v" is standard for .mp4}; * - unpacking operator , which converts the string "mp4v" -> ['m','p','4','v']
        fps,
        (width,height),
    )


    start_time = time.time()
    frame_count = 0


    while True:
        ret, frame = cap.read()
        if not ret:                         # when video ends, ret = false
            break
        image = Image.fromarray(
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )
        )
        image = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(image)
        output = output.squeeze(0)
        output = output.clamp(0,1)
        output = output.detach()
        output = transforms.ToPILImage()(
            output.cpu()
        )
        output = cv2.cvtColor(
            np.array(output),           # PILImg is converted into numpy array
            cv2.COLOR_RGB2BGR           # opencv can understand numpy array , not PILImg
        )
        output = cv2.resize(
            output,
            (width,height)
        )
        out.write(output)               # .write() expects numpy array 
        frame_count += 1
        progress = (frame_count/total_frames)*100

        if frame_count%30 == 0:                             # why 30 ? because 30fps video
            print(f"\rProgress: {progress:.2f}%",end="")



    cap.release()       # closes input video
    out.release()       # saves output video
    end_time = time.time()
    total_time = end_time - start_time
    processing_fps = frame_count / total_time

    print("Done")
    print(f"Output Resolution: {width}x{height}")
    print("\n========== Stats ==========")
    print(f"Frames Processed : {frame_count}")
    print(f"Input FPS        : {fps:.2f}")
    print(f"Time Taken       : {total_time:.2f} sec")
    print(f"Processing FPS   : {processing_fps:.2f}")

    return{
        "frames": frame_count,
        "fps": fps,
        "time": total_time,
        "processing_fps": processing_fps
    }
