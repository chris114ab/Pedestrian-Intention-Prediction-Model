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
import copy

def evaluate_model(model, threshold, data_loader):
    model.eval()
    true_labels = []
    pred_probs = []
    loss = []
    with torch.no_grad():
        for batch in data_loader:
            output = model(batch)
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
      return {"context": list(frames), "label": label, "ped_frames": list(ped_frames), "pose": pose , "bbox": bbox}

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
        return output
    

context, ped_frames, extra = False, False, False
if "con" in sys.argv[1]:
    context = True
if "ped" in sys.argv[1]:
    ped_frames = True
if "ext" in sys.argv[1]:
    extra = True

num_epochs = int(sys.argv[2])
batch_size = int(sys.argv[3])
learning_rate = float(sys.argv[4])
dropout_rate = float(sys.argv[5])
output_path = "/nfs/" + sys.argv[6]
data = str(sys.argv[7])
# if len(sys.argv)==7:
#     dropout_rate = float(sys.argv[6])

model = CombinedTimesformer(context, ped_frames, extra, dropout_rate=dropout_rate)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)


data_path = "/nfs/"+ data+"s_data.pickle"
test_data_path = "/nfs/"+ data+"s_test.pickle"
dataset = PIE_dataset(data_path)
test_dataset = PIE_dataset(test_data_path)
_, val_set = torch.utils.data.random_split(test_dataset, [len(test_dataset)-20, 20])
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
threshold = 0.5
scores = []
best_valscore = 0
best_model = None

print(f"Current Learning Rate: {scheduler.get_last_lr()}")  
for epoch in range(num_epochs):
    model.train()
    true_labels = []
    pred_probs = []
    losses = []
    for batch in data_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = nn.BCEWithLogitsLoss()(output, batch["label"].float().unsqueeze(1))
        loss.backward()
        losses.append(loss.item())
        pred_probs.extend(torch.sigmoid(output).detach().cpu().numpy() > 0.5)
        true_labels.extend(batch["label"].detach().numpy())
        optimizer.step()

    acc_score = accuracy_score(true_labels, pred_probs)
    avg_loss = np.mean(losses)
    eval_score = evaluate_model(model, threshold, val_loader)
    
    if eval_score[1] > best_valscore:
        print("best model yet")
        best_valscore = eval_score[1]
        best_model = copy.deepcopy(model)

    print(f"training loss:{avg_loss} accuracy: {acc_score}")
    print(f"validation loss:{eval_score[5]} accuracy: {eval_score[1]}")
    scores.append(eval_score[:5])
    scheduler.step()

test_score = evaluate_model(best_model, threshold, test_loader)
test = f"Final : F1 Score: {test_score[0]}, Accuracy: {test_score[1]}, Precision: {test_score[2]}, Recall: {test_score[3]}, ROC AUC: {test_score[4]} Average Loss: {test_score[5]}"
print(test)
probs = ",".join([str(x) for x in test_score[6]])
true = ",".join([str(x) for x in test_score[7]])
print(probs)
print(true)

os.mkdir(output_path) 
torch.save(model.state_dict(), output_path + "/model.pt")
with open(output_path + "/scores.txt", "w") as f:
    f.write(str(scores))
    f.write(str(test) + "\n")
    f.write(str(probs) + "\n")   
    f.write(str(true) + "\n")
