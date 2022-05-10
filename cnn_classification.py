from builtins import len

import torch
import torchvision.transforms as transforms
import torch.nn.init
from PIL import Image

# parameters
from sklearn import metrics

learning_rate = 0.01
training_epochs = 30
batch_size = 50

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import os

train_dir = 'train'
test_dir = 'test'
train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)


class CellsDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, label, dir, mode='train', transform=None):
        self.file_list = file_list
        self.dir = dir
        self.mode = mode
        self.transform = transform
        # if self.mode == 'train':
        self.label = label

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, self.file_list[idx]))
        if self.transform:
            img = self.transform(img)
        if self.mode == 'train':
            img = img.numpy()
            return img.astype('float32'), self.label
        else:
            img = img.numpy()
            return img.astype('float32'), self.file_list[idx]


class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 784, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(784, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = torch.nn.Linear(16384, 32, bias=True)
        self.fc2 = torch.nn.Linear(32, 7, bias=True)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.fc2(out)
        return out


# (568, 768)

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

cyl_files = [tf for tf in train_files if 'cyl' in tf]
inter_files = [tf for tf in train_files if 'inter' in tf]
mod_files = [tf for tf in train_files if 'mod' in tf]
let_files = [tf for tf in train_files if 'let' in tf]
super_files = [tf for tf in train_files if 'super' in tf]
para_files = [tf for tf in train_files if 'para' in tf]
svar_files = [tf for tf in train_files if 'svar' in tf]

cyls = CellsDataset(cyl_files, 0, train_dir, transform=data_transform)
inters = CellsDataset(inter_files, 1, train_dir, transform=data_transform)
mods = CellsDataset(mod_files, 2, train_dir, transform=data_transform)
lets = CellsDataset(let_files, 3, train_dir, transform=data_transform)
supers = CellsDataset(super_files, 4, train_dir, transform=data_transform)
paras = CellsDataset(para_files, 5, train_dir, transform=data_transform)
svars = CellsDataset(svar_files, 6, train_dir, transform=data_transform)

cells = torch.utils.data.ConcatDataset([cyls, inters, mods, lets, supers, paras, svars])

# dataset loader
data_loader = torch.utils.data.DataLoader(dataset=cells, batch_size=batch_size, shuffle=True, drop_last=True)

# instantiate CNN model
model = CNN().to(device)



# define cost/loss & optimizer
criterion = torch.nn.CrossEntropyLoss().to(device)  # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train my model
total_batch = len(data_loader)
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

print('Learning Finished!')

cyl_files = [tf for tf in test_files if 'cyl' in tf]
inter_files = [tf for tf in test_files if 'inter' in tf]
mod_files = [tf for tf in test_files if 'mod' in tf]
let_files = [tf for tf in test_files if 'let' in tf]
super_files = [tf for tf in test_files if 'super' in tf]
para_files = [tf for tf in test_files if 'para' in tf]
svar_files = [tf for tf in test_files if 'svar' in tf]

cyls = CellsDataset(cyl_files, 0, test_dir, transform=data_transform)
inters = CellsDataset(inter_files, 1, test_dir, transform=data_transform)
mods = CellsDataset(mod_files, 2, test_dir, transform=data_transform)
lets = CellsDataset(let_files, 3, test_dir, transform=data_transform)
supers = CellsDataset(super_files, 4, test_dir, transform=data_transform)
paras = CellsDataset(para_files, 5, test_dir, transform=data_transform)
svars = CellsDataset(svar_files, 6, test_dir, transform=data_transform)

cells_test = torch.utils.data.ConcatDataset([cyls, inters, mods, lets, supers, paras, svars])

# dataset loader
data_loader = torch.utils.data.DataLoader(dataset=cells_test, batch_size=batch_size, shuffle=True, drop_last=True)

test_y = []
predicted = []

import numpy as np

for X, Y in data_loader:
    X = X.to(device)
    Y = Y.to(device)
    print(len(X), len(Y))
    optimizer.zero_grad()
    hypothesis = model(X)
    cost = criterion(hypothesis, Y)
    y_true = Y.cpu().detach().numpy()

    for i in range(batch_size):
        test_y.append(y_true[i])
        if np.array(hypothesis[i].cpu().detach().numpy())[Y[i].cpu().detach().numpy()] == np.max(
            np.array(hypothesis[i].cpu().detach().numpy())):
            predicted.append(y_true[i])
        else:
            for k in range(len(np.array(hypothesis[i].cpu().detach().numpy()))):
                tmp = np.array(hypothesis[i].cpu().detach().numpy())
                if tmp[k] == np.max(tmp):
                    predicted.append(k)
                    break

    # cost.backward()
    # optimizer.step()

test_y = np.array(test_y)
predicted = np.array(predicted)
print(test_y)
print(predicted)

print("Precision:", round(metrics.precision_score(test_y, predicted, average="micro"), 3),
      "Recall:", round(metrics.recall_score(test_y, predicted, average="micro"), 3),
      "F1-score:", round(metrics.f1_score(test_y, predicted, average="micro"), 3))
