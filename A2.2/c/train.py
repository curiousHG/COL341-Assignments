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

train_data = sys.argv[1]
test_data = sys.argv[2]
model_path = sys.argv[3]
loss_path = sys.argv[4]
acc_path = sys.argv[5]

img_transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
train_dataset = ImageDataset(data_csv = train_data, train=True, img_transform=img_transforms)
test_dataset = ImageDataset(data_csv = test_data, train=True, img_transform=img_transforms)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 200
epochs = 5
l_r = 1e-4
NUM_WORKERS = 20

torch.manual_seed(51)
model = NN(512, 10, 0.2)
model.to(device)
loss = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = l_r)

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=False, num_workers = NUM_WORKERS)
test_loader = DataLoader(dataset = test_dataset, batch_size = len(test_dataset), shuffle=False, num_workers = NUM_WORKERS)

loss_vals = []
accs = []

for epoch in range(epochs):
    train_loss = 0
    accu_train = 0
    for batch in train_loader:
        batch_x, batch_y = batch["images"].to(device), batch["labels"].to(device)
        y_hat = model.forward(batch_x)
        loss_val = loss(y_hat, batch_y)
        train_loss+=loss_val.item()
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
    for batch in test_loader:
        batch_x, batch_y = batch["images"].to(device), batch["labels"].to(device)
        y_hat = model.forward(batch_x)
        predictions = y_hat.argmax(dim=1, keepdim=True).squeeze()
        correct = (predictions == batch_y).sum().item()
        acc = correct / len(batch_y)
        accs.append(acc)

    loss_vals.append(train_loss/len(train_loader))

torch.save(model.state_dict(), model_path)
np.savetxt(acc_path, accs)
np.savetxt(loss_path, loss_vals)