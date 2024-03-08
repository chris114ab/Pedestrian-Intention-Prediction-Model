from ml import Movenet
import pickle
# import random
movenet = Movenet('movenet_thunder')

def detect(ped_frames, inference_count=3):
  ped_pose = []
  for input_tensor in ped_frames:
    # Detect pose using the full input image
    movenet.detect(input_tensor, reset_crop_region=True)
    # Repeatedly using previous detection result to identify the region of
    # interest and only croping that region to improve detection accuracy
    for _ in range(inference_count - 1):
      person = movenet.detect(input_tensor,
                              reset_crop_region=False)
    ped_pose.append(person)
  return ped_pose

with open('data/train_data.pickle', 'rb') as f:
    data = pickle.load(f)

print(len(data["ped_frames"]))

counter =0
ped_pose = []
for i in data["ped_frames"]:
    pose = detect(i)
    ped_pose.append(pose)
    print(counter)
    counter+=1

data["ped_pose"] = ped_pose

with open('data/train_data.pickle', 'wb') as f:
    pickle.dump(data, f)
print("done")
