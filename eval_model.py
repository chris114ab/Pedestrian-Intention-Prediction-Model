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


def evaluate_model(model, threshold, data_loader):
    model.eval()
    true_labels = []
    pred_probs = []
    loss = []
    counter = 0
    with torch.no_grad():
        for batch in data_loader:
            output = model(batch)
            curr_loss = nn.BCEWithLogitsLoss()(output, batch["label"].float().unsqueeze(1))
            loss.append(curr_loss)
            pred_probs.extend(torch.sigmoid(output).detach().cpu().numpy())
            true_labels.extend(batch["label"].detach().numpy())
            print(str(counter)  + " : " + str(curr_loss))
            counter += 1

    loss = np.mean(loss)
    pred_labels = (np.array(pred_probs) > threshold).astype(int)
    true_labels = np.array(true_labels)
    f1 = f1_score(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    roc_auc = roc_auc_score(true_labels, pred_probs)
    
    return f1, accuracy, precision, recall, roc_auc, loss, pred_probs, true_labels

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


class PIE_dataset(Dataset):
    def __init__(self, data_path, transform=None):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.data = self.data

    def __len__(self):
        return len(self.data["labels"])

    def __getitem__(self, idx):
      pose = unload_person(self.data["ped_pose"][idx])
      bbox = self.data["bbox"][idx].flatten()
      label = self.data["labels"][idx]
      return {"context": list(self.data["frames"][idx]), "label": label, "ped_frames": list(self.data["ped_frames"][idx]), "pose": pose , "bbox": bbox}


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class CombinedTimesformer(nn.Module):
    def __init__(self, context, ped_frames, extra):
        super(CombinedTimesformer, self).__init__()
        model_name = "facebook/timesformer-base-finetuned-k400"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        feature_count = 00
        if context:
            feature_count +=768
            self.context_model = TimesformerForVideoClassification.from_pretrained(model_name)
            self.context_model.classifier = Identity()
        if ped_frames:
            feature_count +=768
            self.ped_model = TimesformerForVideoClassification.from_pretrained(model_name)
            self.ped_model.classifier = Identity()
        if extra:
            feature_count += 200
            self.extra_model = torch.nn.Linear(440, 200)
        
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
            feature_vector.append(self.ped_model(**ped_frames).logits)
        if hasattr(self, 'extra_model'):
            extras = torch.cat((x["pose"], x["bbox"]),dim=1).float()
            feature_vector.append(self.extra_model(extras))


        combined_feature_vector = torch.cat(feature_vector, dim=1)
        output = self.classifier(combined_feature_vector)
        return output
    

# model_path = sys.argv[1]
model_path = "/Users/chris/Documents/masters/training_container/data/combinelr-plz/model.pt"
model = CombinedTimesformer(context=True, ped_frames=False, extra=False)
model.load_state_dict(torch.load(model_path))
# data_path = sys.argv[2]
data_path = "/Users/chris/Documents/masters/training_container/data/unseen_data.pickle"
dataset = PIE_dataset(data_path)
loader = DataLoader(dataset, batch_size=10, shuffle=True)
f1, accuracy, precision, recall, roc_auc, loss, pred_probs, true_labels = evaluate_model(model, 0.5, loader)
# print(f1, accuracy, precision, recall, roc_auc, loss)



