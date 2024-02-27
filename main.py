import torch
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
from ml import Movenet
movenet = Movenet('movenet_thunder')

def detect(ped_frames, inference_count=3):
  ped_pose = []
  for input_tensor in ped_frames:
    # Detect pose using the full input image
    movenet.detect(input_tensor, reset_crop_region=True)
    # Repeatedly using previous detection result to identify the region of
    # interest and only croping that region to improve detection accuracy
    for _ in range(inference_count - 1):
      person = movenet.detect(input_tensor,
                              reset_crop_region=False)
    ped_pose.append(person)
  return ped_pose

# normalise and combien thre stream into one
def norm_combine(data1,data2,data3):
  data1 = np.mean(data1,axis=-1, keepdims=True)
  data2 = np.mean(data2,axis=-1, keepdims=True)
  data3 = np.mean(data3,axis=-1, keepdims=True)
  # Concatenate the normalized inputs along the last axis
  input_combined = np.concatenate([data1, data2, data3], axis=-1)
  min_val = np.min(input_combined)
  max_val = np.max(input_combined)
    
  normalized_arr = (input_combined - min_val) / (max_val - min_val)
  return normalized_arr

# combine the person and bbox data
def person_bbox_data(persons,bbox):
      data = np.empty((8,21,3))
      for frame,person in enumerate(persons):
          for index, point in enumerate(person.keypoints):
              data[frame,index]=[point.coordinate.x,point.coordinate.y,point.score]
          data[frame,17:] = [[x,0,0] for x in bbox[frame]]
      data=np.expand_dims(data, axis=0)
      expanded_array = np.zeros((8,224,224,3))
      expanded_array[:, :data.shape[1], :data.shape[2], :] = data
      return expanded_array

def unload(input):
  length = len(input[0])
  videos = [
      [
        tensor[frame_index, :, :, :]
        for tensor in input
      ]
      for frame_index in range(length)
    ]
  return videos

def eval_func(training_model,threshold,validate_loader,processor):
    val_video,val_label =unload(next(iter(validate_loader)))
    val_input = processor(val_video, return_tensors="pt")
    output = training_model(**val_input)
    predicted_label = torch.sigmoid(output.logits) > threshold
    val_label_np = val_label.numpy().astype(int)
    predicted_label_np = predicted_label.numpy().astype(int)
    f1 = f1_score(val_label_np, predicted_label_np)
    print(f1)
    return f1

class PIE_dataset(Dataset):
    def __init__(self, data_path, label_path, ped_path=None, bb_path=None, transform=None):
        self.transform = None
        with open(data_path, "rb") as file:
          # Load the data from the pickle file
          self.data = pickle.load(file)
        with open(label_path, "rb") as file:
          # Load the data from the pickle file
          self.labels = pickle.load(file)
        if ped_path:
          with open(ped_path, "rb") as file:
            # Load the data from the pickle file
            self.ped_frames = pickle.load(file)
        if bb_path:
          with open(bb_path, "rb") as file:
            # Load the data from the pickle file
            self.bbox = pickle.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
      ped_pose = detect(self.ped_frames[idx])
      pose_bb_data = person_bbox_data(ped_pose,self.bbox[idx])
      label = any(i for i in self.labels[idx])
      combine = norm_combine(self.data[idx],self.ped_frames[idx],pose_bb_data)
      return {"video":list(self.data[idx]), "label":label, "bbox":self.bbox[idx], "data":pose_bb_data, "combine":list(combine)}




##############################
batch_size = 5
model_name = "facebook/timesformer-base-finetuned-k400"
frame_path = str(os.getcwd()) + "/data/111_frames.pickle"
label_path = str(os.getcwd()) + "/data/111_label.pickle"
ped_path = str(os.getcwd()) + "/data/ped_frame.pickle"
bb_path = str(os.getcwd()) + "/data/bounding_boxes.pickle"

train_test_split = [100,11]
# try different epochs
# tensorboard to visualise validation loss
# 80/20 split for 
threshold = 0.5
output_path = "100_sample_timesformer"
##############################

print(os.getenv('TFTEST_ENV_VAR'))
model = TimesformerForVideoClassification.from_pretrained(model_name)

# change this for abaltion study
# Machine learning model
# SVM
# Random Forest
model.classifier = torch.nn.Linear(model.classifier.in_features, 1)


processor = AutoImageProcessor.from_pretrained(model_name)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

full_dataset = PIE_dataset(frame_path,label_path,ped_path=ped_path,bb_path=bb_path,transform=None)
train_dataset, test_dataset = random_split(full_dataset, train_test_split)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validate_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


scores=[]
num_epochs=1
counter = 1
for epoch in range(num_epochs):
    print("epoch")
    for batch in train_loader:
        # videos,labels = unload(batch)
        labels = torch.tensor([[x] for x in batch["label"]]).to(torch.float32)
        inputs = processor(list(unload(batch["combine"])), return_tensors="pt")
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        print(counter)
        counter+=1
    scores.append(eval_func(model))

model.save_pretrained(output_path)
# write scores to file
file = open("scores.txt", "w")
file.write(str(scores))
file.close()

# AUC
# f1 score