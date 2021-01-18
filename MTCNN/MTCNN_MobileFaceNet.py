import sys
import os
sys.path.append(os.path.join(sys.path[0], 'MTCNN'))
import argparse
from utils.util import *
from MTCNN import create_mtcnn_net
from face_model import MobileFaceNet
from deal_feature_vector import *

def cosin_dist(a, b):
    ma = np.linalg.norm(a)
    mb = np.linalg.norm(b)
    # Cosine Similarity
    sim = (np.matmul(a, b)) / (ma * mb)

    return sim


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument('-img', '--img', help='upload image', default='images/Howard.jpg', type=str)
    parser.add_argument('-th', '--threshold', help='threshold score to decide identical faces', default=60, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true",
                        default=False)
    parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true", default=False)
    parser.add_argument("-c", "--score", help="whether show the confidence score", action="store_true", default=True)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    detect_model = MobileFaceNet(512).to(device)  # embeding size is 512 (feature vector)
    detect_model.load_state_dict(torch.load('Weights/MobileFace_Net.pt', map_location=lambda storage, loc: storage))
    print('MobileFaceNet face detection model generated')
    detect_model.eval()

    path = "images/Howard.jpg"
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

    # img = cv2.imread(path)
    bboxes, landmarks = create_mtcnn_net(img, 32, device, p_model_path='MTCNN/weights/pnet_Weights.pt',
                                         r_model_path='MTCNN/weights/rnet_Weights',
                                         o_model_path='MTCNN/weights/onet_Weights.pt')

    faces = Face_alignment(img, default_square=True, landmarks=landmarks)
    test_transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    for img in faces:
        emb = detect_model(test_transform(img).to(device).unsqueeze(0))
    for x in emb:
        a = x
        a = a.detach().numpy()

    get_feature_vector(detect_model) #得到feature_vector.csv
    columns = extract_vector_fromcsv()
    columns = [eval(column) for column in columns]
    for column in columns:
        column = np.array(column)
        cos = cosin_dist(a, column)
        print(cos)
