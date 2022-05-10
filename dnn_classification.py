import random
from builtins import len

import torch
import torchvision.transforms as transforms
import torch.nn.init
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# parameters
from sklearn import metrics

learning_rate = 0.0001
training_epochs = 300
batch_size = 50

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import os

data_dir = 'data/Cancerous cell smears'
data_files = os.listdir(data_dir)


def prepare_10fold(data_files):
    eind = 0
    random.shuffle(data_files)
    fold_size = int(len(data_files) / 10)
    for fold in range(0, 10):
        sind = eind
        eind = sind + fold_size
        train_pair = data_files[0:sind] + data_files[eind:len(data_files)]
        test_pair = data_files[sind:eind]
        yield (train_pair, test_pair)


data_10fold = prepare_10fold(data_files)


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


# (568, 768)

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

input_size = 3 * 256 * 256
output_size = 7
hidden_size = [512, 256, 64]


class DNN(torch.nn.Module):

    def __init__(self):
        super(DNN, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size[0])
        self.linear2 = torch.nn.Linear(hidden_size[0], hidden_size[1])
        self.linear3 = torch.nn.Linear(hidden_size[1], hidden_size[2])
        self.linear4 = torch.nn.Linear(hidden_size[2], output_size)

    def forward(self, xb):
        out = xb.view(xb.size(0), -1)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        return out


fold_results = []
precision = []
recall = []
f1 = []
idx = 0
auc = [0, 0, 0, 0, 0, 0, 0]
auc_str = ['', '', '', '', '', '', '']

for item in data_10fold:
    # print(len(item[0]), len(item[1]))
    idx = idx + 1

    train_files = item[0]
    test_files = item[1]

    cyl_files = [tf for tf in train_files if 'cyl' in tf]
    inter_files = [tf for tf in train_files if 'inter' in tf]
    mod_files = [tf for tf in train_files if 'mod' in tf]
    let_files = [tf for tf in train_files if 'let' in tf]
    super_files = [tf for tf in train_files if 'super' in tf]
    para_files = [tf for tf in train_files if 'para' in tf]
    svar_files = [tf for tf in train_files if 'svar' in tf]

    cyls = CellsDataset(cyl_files, 0, data_dir, transform=data_transform)
    inters = CellsDataset(inter_files, 1, data_dir, transform=data_transform)
    mods = CellsDataset(mod_files, 2, data_dir, transform=data_transform)
    lets = CellsDataset(let_files, 3, data_dir, transform=data_transform)
    supers = CellsDataset(super_files, 4, data_dir, transform=data_transform)
    paras = CellsDataset(para_files, 5, data_dir, transform=data_transform)
    svars = CellsDataset(svar_files, 6, data_dir, transform=data_transform)

    cells = torch.utils.data.ConcatDataset([cyls, inters, mods, lets, supers, paras, svars])
    data_loader = torch.utils.data.DataLoader(dataset=cells, batch_size=batch_size, shuffle=True, drop_last=True)
    model = DNN().to(device)

    # print(model)

    # define cost/loss & optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)  # Softmax is internally computed.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train my model
    total_batch = len(data_loader)
    # print('Learning started. It takes sometime.')
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

    # print('Learning Finished!')

    cyl_files = [tf for tf in test_files if 'cyl' in tf]
    inter_files = [tf for tf in test_files if 'inter' in tf]
    mod_files = [tf for tf in test_files if 'mod' in tf]
    let_files = [tf for tf in test_files if 'let' in tf]
    super_files = [tf for tf in test_files if 'super' in tf]
    para_files = [tf for tf in test_files if 'para' in tf]
    svar_files = [tf for tf in test_files if 'svar' in tf]

    cyls = CellsDataset(cyl_files, 0, data_dir, transform=data_transform)
    inters = CellsDataset(inter_files, 1, data_dir, transform=data_transform)
    mods = CellsDataset(mod_files, 2, data_dir, transform=data_transform)
    lets = CellsDataset(let_files, 3, data_dir, transform=data_transform)
    supers = CellsDataset(super_files, 4, data_dir, transform=data_transform)
    paras = CellsDataset(para_files, 5, data_dir, transform=data_transform)
    svars = CellsDataset(svar_files, 6, data_dir, transform=data_transform)

    cells_test = torch.utils.data.ConcatDataset([cyls, inters, mods, lets, supers, paras, svars])

    # dataset loader
    data_loader = torch.utils.data.DataLoader(dataset=cells_test, batch_size=batch_size, shuffle=True, drop_last=True)

    test_y = []
    predicted = []

    import numpy as np

    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)
        # print(len(X), len(Y))
        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)

        y_true = Y.cpu().detach().numpy()
        _, results = torch.max(hypothesis.data, 1)
        print(results)

        test_y.extend(y_true)
        predicted.extend(results.cpu().detach())

    test_y = np.array(test_y)
    predicted = np.array(predicted)

    for u in range(7):
        fpr, tpr, thresholds = metrics.roc_curve(test_y, predicted, pos_label=u)
        print("auc at " + str(u), metrics.auc(fpr, tpr))
        auc[u] += round(metrics.auc(fpr, tpr), 2)
        auc_str[u] = auc_str[u] + " & " + str(round(metrics.auc(fpr, tpr), 2))

    # print("Precision:", round(metrics.precision_score(test_y, predicted, average="micro"), 3),
    #       "Recall:", round(metrics.recall_score(test_y, predicted, average="micro"), 3),
    #       "F1-score:", round(metrics.f1_score(test_y, predicted, average="micro"), 3))
    res = str(idx) + " & " + str(round(metrics.precision_score(test_y, predicted, average="micro"), 2)) + " & " + \
          str(round(metrics.precision_score(test_y, predicted, average="micro"), 2)) + " & " + \
          str(round(metrics.precision_score(test_y, predicted, average="micro"), 2))
    fold_results.append(res)
    precision.append(round(metrics.precision_score(test_y, predicted, average="micro"), 2))
    recall.append(round(metrics.recall_score(test_y, predicted, average="micro"), 2))
    f1.append(round(metrics.f1_score(test_y, predicted, average="micro"), 2))

auc = np.array(auc)
print(auc / 10)

for s in auc_str:
    print(s)


print("Precision:", np.average(np.array(precision)), "Recall:", np.average(np.array(recall)), "F1-score:",
      np.average(np.array(f1)))

for res in fold_results:
    print(res)
