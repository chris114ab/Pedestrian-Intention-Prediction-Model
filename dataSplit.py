import pickle
import os

frame_path = str(os.getcwd()) + "/data/111_frames.pickle"
label_path = str(os.getcwd()) + "/data/111_label.pickle"
ped_path = str(os.getcwd()) + "/data/ped_frame.pickle"
bb_path = str(os.getcwd()) + "/data/bounding_boxes.pickle"

with open(frame_path, "rb") as file:
    frames = pickle.load(file)
with open(label_path, "rb") as file:
    labels = pickle.load(file)
with open(ped_path, "rb") as file:
    ped_frames = pickle.load(file)
with open(bb_path, "rb") as file:
    bbox = pickle.load(file)

crossing = []
not_crossing = []


for index, item in enumerate(labels):
    label = any(i for i in item)
    if label:
        crossing.append(index)
    else:
        not_crossing.append(index)
    

train_test_split = [80,20]
train_crossing = int((train_test_split[0]/100)*len(crossing))
test_crossing = len(crossing) - train_crossing
train_not_crossing = int((train_test_split[0]/100)*len(not_crossing))
test_not_crossing = len(not_crossing) - train_not_crossing

data = {}

for i in range(train_crossing+train_not_crossing):
    data["video"] = frames[i]
    data["label"] = labels[i]
    data["ped_frame"] = ped_frames[i]
    data["bbox"] = bbox[i]

with open("train_data.pickle", "wb") as file:
    pickle.dump(data, file)