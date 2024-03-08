import torch
import sys
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from transformers import AutoImageProcessor, TimesformerForVideoClassification
import torch.nn as nn
import numpy as np
from PIL import Image
import pickle
from torchvision import transforms

def evaluate_model(model, threshold, data_loader):
    model.eval()
    true_labels = []
    pred_probs = []

    with torch.no_grad():
        for batch in data_loader:
            val_video = unload(batch["data"])
            val_input = processor(val_video, return_tensors="pt", padding=True, max_length=100, truncation=True)
            output = model(**val_input)
            pred_probs.extend(torch.sigmoid(output.logits).cpu().numpy())
            true_labels.extend(batch["label"])

    pred_labels = (np.array(pred_probs) > threshold).astype(int)
    true_labels = np.array(true_labels)

    f1 = f1_score(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    roc_auc = roc_auc_score(true_labels, pred_probs)

    return f1, accuracy, precision, recall, roc_auc


def custom_transforms(data):
  transformed_clip = []
  for frame in data:
      # Convert the NumPy array frame to a PIL Image
      pil_img = Image.fromarray(frame)
      # Apply the transformation
      transformed_img = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)(pil_img)
      # Convert back to NumPy array if necessary
      transformed_frame = np.array(transformed_img)
      transformed_clip.append(transformed_frame)
  return np.array(transformed_clip)

# Define custom dataset class
class PIE_dataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.transform =  transforms.RandomApply([custom_transforms], p=0.5)
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.data = self.data

    def __len__(self):
        return len(self.data["labels"])

    def __getitem__(self, idx):
      pose_bb_data = person_bbox_data(self.data["ped_pose"][idx],self.data["bbox"][idx])
      label = any(i for i in self.data["labels"][idx])
      combine = norm_combine(self.data["frames"][idx],self.data["ped_frames"][idx],pose_bb_data)
      data = self.transform(combine)
      return {"data": list(data), "label": label}

# normalise and combine the stream into one
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

# Define model evaluation function
def eval_func(training_model, threshold, validate_loader, processor):
    training_model.eval()
    f1_scores = []
    losses = []
    with torch.no_grad():
        for batch in validate_loader:
            val_video = unload(batch["data"])
            val_input = processor(val_video, return_tensors="pt")
            output = training_model(**val_input)
            predicted_label = torch.sigmoid(output.logits) > threshold
            val_label_np = batch["label"].numpy()
            predicted_label_np = predicted_label.numpy().astype(int)
            f1 = f1_score(val_label_np, predicted_label_np)
            f1_scores.append(f1)
            loss = nn.BCELoss()(torch.flatten(torch.sigmoid(output.logits)), batch["label"].float())
            losses.append(loss.item())
    avg_f1 = np.mean(f1_scores)
    avg_loss = np.mean(losses)
    return avg_f1, avg_loss

# Define training parameters
model_name = "facebook/timesformer-base-finetuned-k400"
threshold = 0.5
batch_size = int(sys.argv[1])
num_epochs = int(sys.argv[2])
output_path = "/nfs/" + sys.argv[3]

# Load pre-trained mod
model = TimesformerForVideoClassification.from_pretrained(model_name)
model.classifier = torch.nn.Linear(model.classifier.in_features, 1)

processor = AutoImageProcessor.from_pretrained(model_name)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Load datasets
train_dataset = PIE_dataset("/nfs/data/train_data.pickle")
validate_dataset = PIE_dataset("/nfs/data/test_data.pickle")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)

scores = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    true_labels = []
    pred_probs = []
    losses = []
    for batch in train_loader:
        labels = torch.tensor(batch["label"]).float().unsqueeze(1)
        inputs = processor(unload(batch["data"]), return_tensors="pt")
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = nn.BCEWithLogitsLoss()(outputs.logits, labels)
        losses.append(loss.item())
        pred_probs.extend(np.array(torch.sigmoid(outputs.logits) > 0.5).astype(int))
        true_labels.extend(batch["label"])
        loss.backward()
        optimizer.step()

    # Evaluate the model after each epoch
    f1 = f1_score(true_labels, pred_probs)
    acc_score = accuracy_score(true_labels, pred_probs)
    precision = precision_score(true_labels, pred_probs)
    recall = recall_score(true_labels, pred_probs)
    roc_auc = roc_auc_score(true_labels, pred_probs)
    avg_loss = np.mean(losses)
    print(f"Epoch {epoch+1}: Training F1 Score: {f1}, Training Loss: {avg_loss}, Accuracy: {acc_score}, Precision: {precision}, Recall: {recall}, ROC AUC: {roc_auc}")
    scores.append({"epoch":epoch,"f1": f1, "loss": avg_loss, "accuracy": acc_score, "precision": precision, "recall": recall, "roc_auc": roc_auc})

# Save the trained model
if output_path:
    test_score = evaluate_model(model, threshold, test_loader)
    model.save_pretrained(output_path)
    # Write scores to file
    with open(output_path + "/training_scores.txt", "w") as file:
        file.write(str(scores))

    with open(output_path + "/test_scores.pickle", "wb") as file:
        file.write(str(test_score))
# test the model
