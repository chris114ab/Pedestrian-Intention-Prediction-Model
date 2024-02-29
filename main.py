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
    batch = next(iter(validate_loader))
    val_video =unload(batch["data"])
    val_input = processor(val_video, return_tensors="pt")
    output = training_model(**val_input)
    predicted_label = torch.sigmoid(output.logits) > threshold
    val_label_np = batch["label"].numpy()
    predicted_label_np = predicted_label.numpy().astype(int)
    f1 = f1_score(val_label_np, predicted_label_np)
    print(f1)
    return f1

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


print("started")
##############################
model_name = "facebook/timesformer-base-finetuned-k400"
# try different epochs
# tensorboard to visualise validation loss
# 80/20 split for 
threshold = 0.5
batch_size = int(sys.argv[1])
num_epochs=int(sys.argv[2])
output_path = "/nfs/" + sys.argv[3]
##############################

# print(os.getenv('TFTEST_ENV_VAR'))
model = TimesformerForVideoClassification.from_pretrained(model_name)

# # change this for abaltion study
# # Machine learning model
# # SVM
# # Random Forest
print(model.classifier.in_features)
model.classifier = torch.nn.Linear(model.classifier.in_features, 1)
print("made model")

processor = AutoImageProcessor.from_pretrained(model_name)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

train_dataset = PIE_dataset("data/train_data.pickle",transform=None)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# validate_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print("loaded data")

# scores=[]
# counter = 1
# for epoch in range(num_epochs):
#     print("epoch")
#     for batch in train_loader:
#         # videos,labels = unload(batch)
#         labels = torch.tensor([[x] for x in batch["label"]]).to(torch.float32)
#         inputs = processor(list(unload(batch["data"])), return_tensors="pt")
#         optimizer.zero_grad()
#         outputs = model(**inputs)
#         loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
#         loss.backward()
#         optimizer.step()
#         print(counter)
#         counter+=1
#     scores.append(eval_func(model,threshold,train_loader,processor))

# model.save_pretrained(output_path)
# # write scores to file
# file = open(output_path + "/training_scores.txt", "w")
# file.write(str(scores))
# file.close()

# AUC
# f1 score