import torch

from face_model import MobileFaceNet


def get_detect_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    detect_model = MobileFaceNet(512).to(device)  # embeding size is 512 (feature vector)
    detect_model.load_state_dict(torch.load('Weights/MobileFace_Net.pt', map_location=lambda storage, loc: storage))
    print('MobileFaceNet face detection model generated')
    detect_model.eval()

    return detect_model

