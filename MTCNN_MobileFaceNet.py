from deal_feature_vector import *
import numpy as np
import pandas as pd


def cosin_dist(a, b):
    ma = np.linalg.norm(a)
    mb = np.linalg.norm(b)
    # Cosine Similarity
    sim = (np.matmul(a, b)) / (ma * mb)

    return sim


def test(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    # img = cv2.imread(path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detect_model = get_detect_model()

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

    columns = extract_vector_fromcsv()  # 取出csv的feature_vector
    columns = [eval(column) for column in columns]
    max_ = 0
    df = pd.read_csv("feature_vector.csv")
    for i, column in enumerate(columns):
        column = np.array(column)
        cos = cosin_dist(a, column)
        # print(cos,df.iloc[i]["name"])
        if max_ < cos:
            max_ = cos
            name = df.iloc[i]["name"]
    if max_ > 0.7:
        print(f'你检测到的人脸是：{name}')
    else:
        print('数据库里没有该人')


if __name__ == '__main__':
    path = "images/test/严敏求.jpg"
    test(path)
