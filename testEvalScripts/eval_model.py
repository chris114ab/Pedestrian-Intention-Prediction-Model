import torch
from torch.utils.data import DataLoader
from utils import evaluate_model
from dataset.pie_dataset import PIE_dataset
from proposed_model import CombinedTimesformer

context, ped_frames, extra = True, True, True
batch_size = 10
model = CombinedTimesformer(context, ped_frames, extra)
with(open("/nfs/cpe-21/model.pt", 'rb')) as f:
    model.load_state_dict(torch.load(f))

paths= ["0.5","1","2"]
for tte in paths:
    data_path = "/nfs/"+ tte+"s_data.pickle"
    test_data_path = "/nfs/"+ tte+"s_test.pickle"
    dataset = PIE_dataset(data_path)
    test_dataset = PIE_dataset(test_data_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    predictions = []
    labels = []
    c = 0
    linear = []
    print(tte)
    data_score = evaluate_model(model, data_loader)
    data = f"Final : F1 Score: {data_score[0]}, Accuracy: {data_score[1]}, Precision: {data_score[2]}, Recall: {data_score[3]}, ROC AUC: {data_score[4]} Average Loss: {data_score[5]}"
    print(data)
    print(data_score[6])
    print(data_score[7])
    test_score = evaluate_model(model, test_loader)
    test = f"Final : F1 Score: {test_score[0]}, Accuracy: {test_score[1]}, Precision: {test_score[2]}, Recall: {test_score[3]}, ROC AUC: {test_score[4]} Average Loss: {test_score[5]}"
    print(test)

data_path = "/nfs/data.pickle"
test_data_path = "/nfs/unseen_data.pickle"
dataset = PIE_dataset(data_path)
test_dataset = PIE_dataset(test_data_path)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
predictions = []
labels = []
c = 0
linear = []
print("0s")
data_score = evaluate_model(model, data_loader)
data = f"Final : F1 Score: {data_score[0]}, Accuracy: {data_score[1]}, Precision: {data_score[2]}, Recall: {data_score[3]}, ROC AUC: {data_score[4]} Average Loss: {data_score[5]}"
print(data)
test_score = evaluate_model(model, test_loader)
test = f"Final : F1 Score: {test_score[0]}, Accuracy: {test_score[1]}, Precision: {test_score[2]}, Recall: {test_score[3]}, ROC AUC: {test_score[4]} Average Loss: {test_score[5]}"
print(test)


