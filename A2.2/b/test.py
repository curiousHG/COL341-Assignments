import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
import sys

class ImageDataset(Dataset):
    
    def __init__(self, data_csv, train = True , img_transform=None):
        self.data_csv = data_csv
        self.img_transform = img_transform
        self.is_train = train
        
        data = pd.read_csv(data_csv, header=None)
        if self.is_train:
            images = data.iloc[:,1:].to_numpy()
            labels = data.iloc[:,0].astype(int)
        else:
            images = data.iloc[:,:].to_numpy()
            labels = None
        
        self.images = images
        self.labels = labels
        print("Total Images: {}, Data Shape = {}".format(len(self.images), images.shape))
        
    def __len__(self):
        """Returns total number of samples in the dataset"""
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        image = np.array(image).astype(np.uint8).reshape((32, 32, 3), order= "F")
        
        if self.is_train:
            label = self.labels[idx]
        else:
            label = -1
        
        image = self.img_transform(image)
        
        sample = {"images": image, "labels": label}
        return sample

class NN(nn.Module):
    def __init__(self,n_h1,n_h2,p):
        super(NN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3,stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(2,stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3,stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 512, 3,stride=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, 2,stride=1)
        self.fc1 = nn.Linear(1024, n_h1)
        self.dropout = nn.Dropout(p)
        self.fc2 = nn.Linear(n_h1, n_h2)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

test_data = sys.argv[1]
model_path = sys.argv[2]
pred_path = sys.argv[3]

NUM_WORKERS = 20

img_transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
test_dataset = ImageDataset(
    data_csv = test_data, 
    train=False, 
    img_transform=img_transforms
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(51)
model = NN(256, 10, 0.2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

test_loader = DataLoader(
    dataset = test_dataset, 
    batch_size = 200, 
    shuffle=False, 
    num_workers = NUM_WORKERS
)
a = np.array([])
for batch in test_loader:
    batch_x, batch_y = batch["images"].to(device), batch["labels"].to(device)
    y_hat = model.forward(batch_x)
    y_hat = torch.argmax(y_hat, dim=1, keepdim=True).squeeze()
    t = y_hat.detach().cpu().numpy()
    a = np.append(a,t)
np.savetxt(pred_path, a)