import cv2
import torch
import numpy as np
from PIL import Image 
import util
import model
from torchvision import transforms
from torch import nn, optim

def main():

    device = torch.device("cuda:0")

    model_load = model.LandmarkDetector()
    checkpoint_path = 'D:\AI_CODE\code_camp_2024\Filter\model/resnet50_fulltrain.pth'
    model_load.load_state_dict(torch.load(checkpoint_path, map_location="cuda:0"))
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    simple_transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


    cap = cv2.VideoCapture(0)  # Mở camera mặc định (0)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        # Chuyển đổi frame OpenCV sang PIL Image
        image_input = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 

        # Thực hiện inference
        image_result = util.inference(model_load=model_load, simple_transform=simple_transform,image_input=image_input)

        # Chuyển đổi lại thành frame OpenCV
        frame_result = cv2.cvtColor(np.array(image_result), cv2.COLOR_RGB2BGR)

        # Hiển thị frame
        cv2.imshow("Face Tracking", frame_result)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn 'q' để thoát
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
