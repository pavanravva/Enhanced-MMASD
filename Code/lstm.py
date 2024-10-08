import os
import pandas as pd

list_folders = [
    '/content/drive/MyDrive/Complete_MMASD_DATA_25_4_2024',

 ]

all_files = []
for n in range(0, len(list_folders)):
  files_names = os.listdir(list_folders[n])
  for m in range(0, len(files_names)):
   files_names[m] = os.path.join(list_folders[n], files_names[m])
   all_files.append(files_names[m])
len(all_files)

append_all_ipyn_checkpoints = []
for n in range(0, len(all_files)):
  if all_files[n].split('/')[-1] == '.ipynb_checkpoints':
    append_all_ipyn_checkpoints.append(all_files[n])

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = pd.read_csv(self.file_list[idx])
        data = data.dropna()
        data = data.iloc[1:]
        columns_values = data.columns
        if data.columns[0] == 'Unnamed: 0':
            data = data.drop('Unnamed: 0', axis=1)

        if len(data) < 180:
              repeat_times = (180 // len(data)) + 1
              data = pd.concat([data] * repeat_times, ignore_index=True)

        data = data[:179]

        data_miss = data.drop(['ASD_Label', 'Action_Label'], axis = 1)
        data_min = data_miss.min()
        data_max = data_miss.max()

        data_norm = (data_miss - data_min)/ (data_max - data_min)
        normalization_data = torch.tensor(data_norm.values, dtype=torch.float)


        data_label = data['Action_Label']

        #print(data_label)
        #if data_label.iloc[0] > 0:
        #    data_label = np.repeat(1, len(data))
        #    data_label = torch.tensor(data_label, dtype=torch.long)
        #else:
        data_label = torch.tensor(data_label.values, dtype=torch.long)
        return data_label, normalization_data


def preprocess(data):
    return ToTensor()(data.values)

file_list = all_files

dataset = CustomDataset(file_list, transform=preprocess)

train_size = int(0.8 * len(dataset))
val_size = (len(dataset) - train_size) // 2
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out

input_size = 75
hidden_size = 64
num_layers = 4
num_classes = 11

model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


import torch
import torch.nn as nn
import torch.optim as optim

model.to(device)

num_epochs = 100
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for labels, inputs in train_loader:

        inputs =  inputs.to(device)
        labels = labels.to(device)
        labels = labels[:,0]

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    train_loss = running_loss / len(train_loader.dataset)

    model.eval()
    val_running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for labels, inputs in val_loader:
            inputs = inputs.to(device)
            labels =  labels.to(device)
            labels = labels[:,0]
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)
            val_running_loss += val_loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_running_loss / len(val_loader.dataset):.4f}, Accuracy: {val_accuracy:.2f}%')

    # Save model checkpoint periodically and at the end
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        torch.save(model.state_dict(), f'/content/drive/MyDrive/New_LSTM_Action_Classification_8_1_2024/lstm_model_epoch_{epoch}.pth')

print("Training complete!")



import torch
from sklearn.metrics import classification_report

model.eval()

all_labels = []
all_predictions = []

# No need to compute gradients for evaluation
with torch.no_grad():
    for labels, videos1 in val_loader:
        videos1,  labels = videos1.to(device),  labels.to(device)
        outputs = model(videos1)
        _, predicted = torch.max(outputs.data, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())





# Generate the classification report
report = classification_report(np.array(all_labels)[:,0], all_predictions, target_names=['Arm_Swing',
                                                                         'Body_pose',
                                                                          'Drumming',
                                                                        'Frog_Pose',
                                                                        'Marcas_Forward',
                                                                         'Marcas_Shaking',
                                                                        'Sing_Clap',
                                                                         'Squat_Pose',
                                                                         'Tree_Pose',
                                                                          'Twist_Pose',
                                                                         'chest_expansion' ])
print(report)



