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
        self.transform =  transforms.RandomApply([custom_transforms], p=0.5)

    def __len__(self):
        return len(self.data["labels"])

    def __getitem__(self, idx):
      pose = unload_person(self.data["ped_pose"][idx])
      bbox = self.data["bbox"][idx].flatten()
      label = self.data["labels"][idx]
      frames,ped_frames = self.transform([self.data["frames"][idx], self.data["ped_frames"][idx]])
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
            self.context_model.classifier = Identity()
        if ped_frames:
            feature_count +=768
            self.ped_model = TimesformerForVideoClassification.from_pretrained(model_name)
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
            feature_vector.append(self.ped_model(**ped_frames).logits)
        if hasattr(self, 'extra_model'):
            extras = torch.cat((x["pose"], x["bbox"]),dim=1).float()
            feature_vector.append(self.extra_model(extras))


        combined_feature_vector = torch.cat(feature_vector, dim=1)
        combined_feature_vector = self.dropout(combined_feature_vector)
        output = self.classifier(combined_feature_vector)
        return combined_feature_vector, output

context, ped_frames, extra = False, False, False
if "con" in sys.argv[1]:
    context = True
if "ped" in sys.argv[1]:
    ped_frames = True
if "ext" in sys.argv[1]:
    extra = True

batch_size = 10
dropout_rate=float(sys.argv[2])
path = "/nfs/" + sys.argv[3]

model = CombinedTimesformer(context, ped_frames, extra, dropout_rate=dropout_rate)

data_path = "/nfs/data.pickle"
test_data_path = "/nfs/unseen_data.pickle"
dataset = PIE_dataset(data_path)
test_dataset = PIE_dataset(test_data_path)
test_set, val_set = torch.utils.data.random_split(test_dataset, [len(test_dataset)-20, 20])
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
scores = []

# load fine-tuned model
with(open(path+"/model.pt", 'rb')) as f:
    model.load_state_dict(torch.load(f))

# apply the model to all the train data
features = []
labels = []
c = 0
linear = []
for batch in data_loader:
    feature, _ = model(batch)
    features.extend(feature.detach().numpy())
    labels.extend(batch["label"])

# fit the classifier on the output of the model

rf = RandomForestClassifier(n_estimators=768)
rf.fit(features, labels)
# evaluate the classifier on the test data
scores = evaluate_model(model,test_loader, rf)
test = f"RF : F1 Score: {scores[0]}, Accuracy: {scores[1]}, Precision: {scores[2]}, Recall: {scores[3]}"
print(test)
print(scores[4])
print(scores[5])

svm_cls = svm.SVC()
svm_cls.fit(features, labels)
scores = evaluate_model(model,test_loader, rf)
test = f"SVM : F1 Score: {scores[0]}, Accuracy: {scores[1]}, Precision: {scores[2]}, Recall: {scores[3]}"
print(test)
print(scores[4])
print(scores[5])

scores = evaluate_model_linear(model,0.5,test_loader)
test = f"Linear : F1 Score: {scores[0]}, Accuracy: {scores[1]}, Precision: {scores[2]}, Recall: {scores[3]}"
print(test)
print(scores[6])
print(scores[7])

