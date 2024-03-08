import torch
import sys
from torch.utils.data import Dataset,DataLoader,random_split
from sklearn.metrics import f1_score
from transformers import AutoImageProcessor, TimesformerForVideoClassification
import torch.nn as nn
import os
import xml.etree.ElementTree as ET
import cv2
import pickle
from PIL import Image
import numpy as np


class PIE_dataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = None
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.data = self.data

    def __len__(self):
        return len(self.data["labels"])

    def __getitem__(self, idx):
      pose_bb_data = person_bbox_data(self.data["ped_pose"][idx],self.data["bbox"][idx])
      label = any(i for i in self.data["labels"][idx])
      combine = norm_combine(self.data["frames"][idx],self.data["ped_frames"][idx],pose_bb_data)
      return {"data":list(combine), "label":label}

