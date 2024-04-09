from torch.utils.data import Dataset
import pickle
from torchvision import transforms
from utils import unload_person

class PIE_dataset(Dataset):
    def __init__(self, data_path,G=True,L=True,E=True,transform=None):
        self.g,self.l,self.e  = G , L ,E
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.transform =  transform

    def __len__(self):
        return len(self.data["labels"])

    def __getitem__(self, idx):
      pose = unload_person(self.data["ped_pose"][idx])
      bbox = self.data["bbox"][idx].flatten()
      label = self.data["labels"][idx]
      frames = self.data["frames"][idx]
      ped_frames = self.data["ped_frames"][idx]

      if self.transform:  
        frames,ped_frames = self.transform([frames, ped_frames])
        
      output = {}
      output["label"] = label
      if self.g:
          output["contex"] = list(frames)
      if self.l:
          output["ped_frames"] = list(ped_frames)
      if self.e:
          output["pose"] = pose
          output["bbox"] = bbox

      return output
