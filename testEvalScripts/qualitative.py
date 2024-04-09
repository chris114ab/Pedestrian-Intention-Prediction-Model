import torch
import os
import sys
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from transformers import AutoImageProcessor, TimesformerForVideoClassification
import torch.nn as nn
import numpy as np
import pickle
from PIL import Image
from sklearn import svm
import cv2
from utils import unload,unload_person,custom_transforms
import matplotlib.pyplot as plt
from utils import evaluate_model,unload,unload_person,custom_transforms
from dataset.pie_dataset import PIE_dataset
from proposed_model import CombinedTimesformer

context, ped_frames, extra = True, False, False

batch_size = 1

model = CombinedTimesformer(context, ped_frames, extra)

data_path = "unseen_data.pickle"
test_data_path = "data/0.5s_data.pickle"
dataset = PIE_dataset(data_path)
test_dataset = PIE_dataset(test_data_path)
data_loader = DataLoader(dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
scores = []

# load fine-tuned model
with(open("model_outputs/cpe-21/model.pt", 'rb')) as f:
    model.load_state_dict(torch.load(f))

model.eval()

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data.shape)

print(sum(p.numel() for p in model.parameters()))

counter = 0

for i in range(50,51):
    print(i)
    cv2.imshow("frame",dataset[i]["ped_frames"][0])
    cv2.waitKey(0)
    for frame in range(8):
        cv2.imshow("frame",dataset[i]["ped_frames"][frame])
        cv2.waitKey(0)



