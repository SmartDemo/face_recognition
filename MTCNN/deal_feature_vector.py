from MTCNN import create_mtcnn_net
import torch
from torchvision import transforms as trans
import time
import multiprocessing
# from time import sleep
from tqdm import *
import csv
import os
import cv2
import numpy as np

from utils.align_trans import Face_alignment


def get_feature_vector(detect_model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    path = "images"
    with open('feature_vector.csv', 'w', encoding='utf-8-sig', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'feature_vector'])

        for i, image_file in enumerate(tqdm(os.listdir(path))):
            # sleep(.01)
            image_path = os.path.join(path, image_file)
            image_name = os.path.basename(image_path).split('.')[0]

            try:
                img1 = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                # cv2.imshow("hello",img1)
                # cv2.waitKey(100)
                bboxes, landmarks = create_mtcnn_net(img1, 32, device, p_model_path='MTCNN/weights/pnet_Weights.pt',
                                                     r_model_path='MTCNN/weights/rnet_Weights',
                                                     o_model_path='MTCNN/weights/onet_Weights.pt')
                faces1 = Face_alignment(img1, default_square=True, landmarks=landmarks)
                for img1 in faces1:
                    emb1 = detect_model(test_transform(img1).to(device).unsqueeze(0))
                for x in emb1:
                    b = x
                    # print(b)
                    b = b.detach().numpy().tolist()
                    writer.writerow([image_name, b])
                # print(f'这是第{i}趟')
            except Exception as e:
                print(f'error:{e}')


def run_multiprocess(detect_model):
    start_time = time.time()
    processes = list()
    for i in range(0, 10):
        p = multiprocessing.Process(target=get_feature_vector, args=(detect_model,))
        processes.append(p)
        p.start()
    for process in processes:
        process.join()
    print('Multiprocessing took {} seconds'.format(time.time() - start_time))


def extract_vector_fromcsv():
    with open('feature_vector.csv', 'rt', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        columns = [row['feature_vector'] for row in reader]
    return columns

# if __name__ == "__main__":
# run_multiprocess()
# extract_vector_fromcsv()
