import torch
import os
import sys
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from torchvision import transforms
import torch.nn as nn
import numpy as np
import pickle
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import cv2

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # For nice heatmap aesthetics


def evaluate_model_linear(model, threshold, data_loader):
    model.eval()
    true_labels = []
    pred_probs = []
    loss = []
    with torch.no_grad():
        for batch in data_loader:
            _,output = model(batch)
            loss.append(nn.BCEWithLogitsLoss()(output, batch["label"].float().unsqueeze(1)))
            pred_probs.extend(torch.sigmoid(output).detach().cpu().numpy())
            true_labels.extend(batch["label"].detach().numpy())

    loss = np.mean(loss)
    pred_labels = (np.array(pred_probs) > threshold).astype(int)
    true_labels = np.array(true_labels)
    f1 = f1_score(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    roc_auc = roc_auc_score(true_labels, pred_probs)
    return f1, accuracy, precision, recall, roc_auc, loss, pred_probs, true_labels



def evaluate_model(model, data_loader,classifier):
    model.eval()
    true_labels = []
    pred_labels = []
    c=0
    with torch.no_grad():
        for batch in data_loader:
            features,_ = model(batch)
            output = classifier.predict(features)
            pred_labels.extend(output)
            true_labels.extend(batch["label"].detach().numpy())

    true_labels = np.array(true_labels)
    f1 = f1_score(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    return f1, accuracy, precision, recall, pred_labels, true_labels

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

def unload_person(pose):
    data = np.zeros((len(pose), 17, 3))
    for frame,person in enumerate(pose):
        for index, point in enumerate(person.keypoints):
            data[frame,index]=[point.coordinate.x,point.coordinate.y,point.score]
    flatten = data.flatten()
    return flatten


def custom_transforms(data):
  data1, data2 = data
  transformed_clip1 = []
  transformed_clip2 = []
  for index in range(len(data1)):
      # Convert the NumPy array frame to a PIL Image
      pil_img = Image.fromarray(data1[index])
      pill_img2 = Image.fromarray(data2[index])
      # Apply the transformation
      transformed_img1 = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)(pil_img)
      transformed_img2 = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)(pill_img2)
      # Convert back to NumPy array if necessary
      transformed_frame1 = np.array(transformed_img1)
      transformed_frame2 = np.array(transformed_img2)
      transformed_clip1.append(transformed_frame1)
      transformed_clip2.append(transformed_frame2)
  return np.array(transformed_clip1), np.array(transformed_clip2)


class PIE_dataset(Dataset):
    def __init__(self, data_path, transform=None):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.data = self.data
        # self.transform =  transforms.RandomApply([custom_transforms], p=0.5)

    def __len__(self):
        return len(self.data["labels"])

    def __getitem__(self, idx):
      pose = unload_person(self.data["ped_pose"][idx])
      bbox = self.data["bbox"][idx].flatten()
      label = self.data["labels"][idx]
      frames =  self.data["frames"][idx]
      ped_frames = self.data["ped_frames"][idx]
    #   frames,ped_frames = self.transform([self.data["frames"][idx], self.data["ped_frames"][idx]])
      return {"context": list(frames), "label": label, "ped_frames": list(ped_frames), "pose": pose ,"bbox":bbox}

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class CombinedTimesformer(nn.Module):
    def __init__(self, context, ped_frames, extra, dropout_rate=0.5):
        super(CombinedTimesformer, self).__init__()
        model_name = "facebook/timesformer-base-finetuned-k400"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        feature_count = 00
        if context:
            feature_count +=768
            self.context_model = TimesformerForVideoClassification.from_pretrained(model_name)
            self.context_model.config.output_attentions = True
            self.context_model.classifier = Identity()
        if ped_frames:
            feature_count +=768
            self.ped_model = TimesformerForVideoClassification.from_pretrained(model_name)
            self.ped_model.config.output_attentions = True
            self.ped_model.classifier = Identity()
        if extra:
            feature_count += 200
            self.extra_model = torch.nn.Linear(440, 200)
        
        self.dropout = nn.Dropout(dropout_rate)  # Add dropout layer
        self.feature_count = feature_count

        self.classifier =torch.nn.Linear(feature_count, 1)
        
    def forward(self, x):
        context = self.processor(unload(x["context"]), return_tensors="pt")
        ped_frames = self.processor(unload(x["ped_frames"]), return_tensors="pt")

        feature_vector = []
        feature_count = 0
        if hasattr(self, 'context_model'):
            feature_vector.append(self.context_model(**context).logits)
           
        if hasattr(self, 'ped_model'):
            ped = self.ped_model(**ped_frames)
            print(len(ped.attentions))
            print(ped.attentions[0].shape)
            feature_vector.append(ped.logits)
        if hasattr(self, 'extra_model'):
            extras = torch.cat((x["pose"], x["bbox"]),dim=1).float()
            feature_vector.append(self.extra_model(extras))


        combined_feature_vector = torch.cat(feature_vector, dim=1)
        combined_feature_vector = self.dropout(combined_feature_vector)
        output = self.classifier(combined_feature_vector)
        return output, ped.attentions[0]

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
# with(open("model_outputs/cpe-21/model.pt", 'rb')) as f:
#     model.load_state_dict(torch.load(f))

# model.eval()
# apply the model to all the train data

# batch = next(iter(data_loader))
# output, attention = model(batch)
# average_attention = torch.mean(attention[1], dim=0)  # Shape [197, 197]


# attention_to_visualize = average_attention.detach().numpy()  # Or a specific row/column of it
# plt.figure(figsize=(10, 8))
# sns.heatmap(attention_to_visualize, cmap='viridis')
# plt.title("Attention Heatmap")
# plt.xlabel("Target Tokens/Patches")
# plt.ylabel("Source Tokens/Patches")
# plt.show()

# print parameters
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data.shape)

print(sum(p.numel() for p in model.parameters()))

counter = 0

# for i in range(50,51):
#     print(i)
#     cv2.imshow("frame",dataset[i]["ped_frames"][0])
#     cv2.waitKey(0)
#     for frame in range(8):
#         cv2.imshow("frame",dataset[i]["ped_frames"][frame])
#         cv2.waitKey(0)

# for batch in data_loader:
#     if counter!=50:
#         counter += 1
#         continue
#     output = model(batch)
#     probs = torch.sigmoid(output).detach().cpu().numpy()
    # print(probs)
    # ped_frames = unload(batch["ped_frames"])
    # context = unload(batch["context"])
    # for frame in range(8):
    #     cv2.imshow("frame",ped_frames[0][frame].numpy())
    #     cv2.waitKey(0)
    # if probs[0]<0.5:
    #     print(counter)
    #     ped_frames = unload(batch["ped_frames"])
    #     context = unload(batch["context"])
    #     for frame in range(8):
    #         cv2.imshow("frame",ped_frames[0][frame].numpy())
    #         cv2.waitKey(0)
    #     break
    # break




