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
from utils import evaluate_model, unload, unload_person, custom_transforms
from dataset.pie_dataset import PIE_dataset
from proposed_model import CombinedTimesformer
    
    
# feature configuiration
context, ped_frames, extra = False, False, False
if "con" in sys.argv[1]:
    context = True
if "ped" in sys.argv[1]:
    ped_frames = True
if "ext" in sys.argv[1]:
    extra = True

# hyperparameters
num_epochs = int(sys.argv[2])
batch_size = int(sys.argv[3])
learning_rate = float(sys.argv[4])
dropout_rate = float(sys.argv[5])
output_path = "/nfs/" + sys.argv[6]
dropout_rate = float(sys.argv[6]) 
data = str(sys.argv[7])

model = CombinedTimesformer(context, ped_frames, extra, dropout_rate=dropout_rate)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)


data_path = "/nfs/"+ data+"s_data.pickle"
test_data_path = "/nfs/"+ data+"s_test.pickle"
dataset = PIE_dataset(data_path,context,ped_frames,extra,transform=custom_transforms)
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
