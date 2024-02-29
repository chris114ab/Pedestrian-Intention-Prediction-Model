import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import random

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

# Define a simple linear model
class LinearModel(nn.Module):
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.fc(x)
        return torch.sigmoid(x)  # Apply sigmoid activation for binary classification

# Assuming you have a dataset X_train and y_train for training
# X_train.shape should be (num_samples, 8, 224, 224, 3)
# y_train.shape should be (num_samples,)

with open('data/train_data.pickle', 'rb') as f:
    data = pickle.load(f)

X_train = []
y_train = []
for i in range(len(data["frames"])): 
    pose_bb_data = person_bbox_data(data["ped_pose"][i],data["bbox"][i])
    label = any(i for i in data["labels"][i])
    combine = norm_combine(data["frames"][i],data["ped_frames"][i],pose_bb_data)
    X_train.append(combine)
    y_train.append(label)
print(y_train)

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Define the model
input_size = 8 * 224 * 224 * 3
model = LinearModel(input_size)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training loop
epochs = 10
batch_size = 32
for epoch in range(epochs):
    for i in range(0, len(data["labels"])):
        inputs = X_train_tensor[i:i+batch_size]
        targets = y_train_tensor[i:i+batch_size]
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets.view(-1, 1))  # Reshape targets to match output shape
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(X_train)}], Loss: {loss.item():.4f}')

# Evaluate the model
# Assuming you have a separate test set X_test and y_test
# X_test.shape and y_test.shape should be similar to X_train and y_train
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# model.eval()  # Set model to evaluation mode
# with torch.no_grad():
#     outputs = model(X_test_tensor)
#     predicted = (outputs >= 0.5).squeeze().cpu().numpy()  # Convert probabilities to binary predictions
#     accuracy = np.mean(predicted == y_test)  # Calculate accuracy
#     print(f'Test Accuracy: {accuracy:.4f}')
