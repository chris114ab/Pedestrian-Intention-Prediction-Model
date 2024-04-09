import torch
import sys
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from utils import evaluate_model
from dataset.pie_dataset import PIE_dataset
from proposed_model import CombinedTimesformer


def evaluate_model_classifier(model, data_loader,classifier):
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

model = CombinedTimesformer(context, ped_frames, extra, dropout_rate=dropout_rate, feature_vector=True)

data_path = "/nfs/data.pickle"
test_data_path = "/nfs/unseen_data.pickle"
dataset = PIE_dataset(data_path,context,ped_frames,extra)
test_dataset = PIE_dataset(test_data_path,context,ped_frames,extra)
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
scores = evaluate_model_classifier(model,test_loader, rf)
test = f"RF : F1 Score: {scores[0]}, Accuracy: {scores[1]}, Precision: {scores[2]}, Recall: {scores[3]}"
print(test)
print(scores[4])
print(scores[5])

svm_cls = svm.SVC()
svm_cls.fit(features, labels)
scores = evaluate_model_classifier(model,test_loader, rf)
test = f"SVM : F1 Score: {scores[0]}, Accuracy: {scores[1]}, Precision: {scores[2]}, Recall: {scores[3]}"
print(test)
print(scores[4])
print(scores[5])

scores = evaluate_model(model,0.5,test_loader)
test = f"Linear : F1 Score: {scores[0]}, Accuracy: {scores[1]}, Precision: {scores[2]}, Recall: {scores[3]}"
print(test)
print(scores[6])
print(scores[7])

