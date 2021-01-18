#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:09:25 2019
Evaluation of MTCNN & Mobilefacenet via Picture

@author: AIRocker
"""

import sys
import os
sys.path.append(os.path.join(sys.path[0], 'MTCNN'))
import argparse
import torch
from torchvision import transforms as trans
from PIL import Image, ImageDraw, ImageFont
from utils.util import *
from utils.align_trans import *
from MTCNN import create_mtcnn_net
from face_model import MobileFaceNet, l2_norm
from facebank import load_facebank, prepare_facebank
import cv2
from scipy.spatial.distance import pdist
import csv
import pathlib
from tqdm import *
from time import sleep 
from deal_feature_vector import *

# Linear Algebra Learning Sequence
# Cosine Similarity
def cosin_dist(a,b):
    ma = np.linalg.norm(a)
    mb = np.linalg.norm(b)  
    # Cosine Similarity
    sim = (np.matmul(a,b))/(ma*mb)

    return sim


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument('-img', '--img', help='upload image', default='images/Howard.jpg', type=str)
    parser.add_argument('-th','--threshold',help='threshold score to decide identical faces',default=60, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true", default= False)
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true", default= False)
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true",default= True )
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    detect_model = MobileFaceNet(512).to(device)  # embeding size is 512 (feature vector)
    detect_model.load_state_dict(torch.load('Weights/MobileFace_Net.pt', map_location=lambda storage, loc: storage))
    print('MobileFaceNet face detection model generated')
    detect_model.eval()


    path = "images/1931女子偶像组合.jpg"
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)

    # img = cv2.imread(path)
    bboxes, landmarks = create_mtcnn_net(img, 32, device, p_model_path='MTCNN/weights/pnet_Weights.pt',
                                         r_model_path='MTCNN/weights/rnet_Weights',
                                         o_model_path='MTCNN/weights/onet_Weights.pt')
    
    faces = Face_alignment(img, default_square = True,landmarks = landmarks)
    test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    for img in faces:
        emb = detect_model(test_transform(img).to(device).unsqueeze(0))
    for x in emb:
        a = x
        a = a.detach().numpy()

    # get_feature_vector(detect_model) #得到feature_vector.csv
    columns = extract_vector_fromcsv()
    columns = [eval(column) for column in columns]
    for column in columns:
        column = np.array(column)
        cos = cosin_dist(a,column)
        print(cos)


        












    # if args.update:
    #     targets, names = prepare_facebank(detect_model, path='facebank', tta=args.tta)
    #     print('facebank updated')
    # else:
    #     targets, names = load_facebank(path='facebank')
    #     print('facebank loaded')
    #     # targets: number of candidate x 512
    # image = cv2.imread(args.img)

    # bboxes, landmarks = create_mtcnn_net(image, 32, device, p_model_path='MTCNN/weights/pnet_Weights.pt',
    #                                     r_model_path='MTCNN/weights/rnet_Weights',
    #                                     o_model_path='MTCNN/weights/onet_Weights.pt')

    # faces = Face_alignment(image, default_square = True,landmarks = landmarks)

    # embs = []


    # test_transform = trans.Compose([
    #                 trans.ToTensor(),
    #                 trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # for img in faces:
    #     if args.tta:
    #         mirror = cv2.flip(img,1)
    #         emb = detect_model(test_transform(img).to(device).unsqueeze(0))
    #         emb_mirror = detect_model(test_transform(mirror).to(device).unsqueeze(0))
    #         embs.append(l2_norm(emb + emb_mirror))
    #     else:
    #         embs.append(detect_model(test_transform(img).to(device).unsqueeze(0)))

    # source_embs = torch.cat(embs)  # number of detected faces x 512
    # diff = source_embs.unsqueeze(-1) - targets.transpose(1, 0).unsqueeze(0) # i.e. 3 x 512 x 1 - 1 x 512 x 2 = 3 x 512 x 2
    # dist = torch.sum(torch.pow(diff, 2), dim=1) # number of detected faces x numer of target faces
    # minimum, min_idx = torch.min(dist, dim=1) # min and idx for each row
    # min_idx[minimum > ((args.threshold-156)/(-80))] = -1  # if no match, set idx to -1
    # score = minimum
    # results = min_idx

    # # convert distance to score dis(0.7,1.2) to score(100,60)
    # score_100 = torch.clamp(score*-80+156,0,100)
    # print(' score:{:.0f}'.format(score_100[0]))





