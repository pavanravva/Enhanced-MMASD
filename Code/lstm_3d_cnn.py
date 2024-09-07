import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
import numpy as np
import torch.nn.functional as F
import pandas as pd

class CustomVideoDataset(Dataset):
    def __init__(self, video_folders_list, num_frames, frame_height, frame_width):

        self.video_files_1 =  video_folders_list

        self.num_frames = num_frames
        self.frame_height = frame_height
        self.frame_width = frame_width

    def __len__(self):
        return len(self.video_files_1)

    def __getitem__(self, idx):

        optical_flow_video = 'Complete_Files_Optical_Together' +  '/' + self.video_files_1[idx] + '.mp4'
        Lstm_Data_Normalized = 'Complete_3D_Skeleton_Data_Star' + '/' + self.video_files_1[idx] + '.csv'

        video_path_1 = optical_flow_video
        lstm_path = Lstm_Data_Normalized

        frames_1 = self._load_frames(video_path_1)
        lstm_data = self._load_dataframe(lstm_path)


        video_action = '_'.join(self.video_files_1[idx].split('_')[:2])
        label = 0
        action_to_label = {
            'processed_Arm': 0, 'processed_bs': 1, 'processed_ce': 2, 'processed_dr': 3,
            'processed_fg': 4, 'processed_mfs': 5, 'processed_ms': 6,
            'processed_sq': 7, 'processed_tw': 8, 'processed_sac': 9, 'processed_tr': 10
        }

        label = action_to_label.get(video_action, label)
        label = torch.tensor(label, dtype=torch.long)

        return frames_1, lstm_data, label

    def _load_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Explicitly convert to grayscale
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))  # Resizing the frame
            frames.append(frame)
            if len(frames) == self.num_frames:
                break
        cap.release()

        # Handle case where video is shorter than num_frames
        while len(frames) < self.num_frames:
            frames.append(np.zeros((self.frame_height, self.frame_width), dtype=np.float32))  # Gray frame

        video_tensor = torch.tensor(np.stack(frames, axis=0)).unsqueeze(1).float() / 255  # Convert to torch tensor and normalize to [0,1]
        video_tensor = video_tensor.permute(1, 0, 2, 3)  # Reorder dimensions to [1, 16, 224, 224]

        return video_tensor

    def _load_dataframe(self, lstm_path):

        df = pd.read_csv(lstm_path)
        df = df.drop(['Action_Label', 'ASD_Label'], axis = 1)
        df_min = df.min().min()
        df_max = df.max().max()

        normalized_data = (df - df_min)/(df_max - df_min)

        data_array = normalized_data.values

        data_tensor = torch.tensor(data_array, dtype=torch.float)
        return data_tensor

video_folder_optical_flow =  'Complete_Files_Optical_Together'
lstm_table_data = 'Complete_3D_Skeleton_Data_Star'

list_files_optical_flow = os.listdir(video_folder_optical_flow)
list_files_lstm_data = os.listdir(lstm_table_data)

list_files_optical_flow[0]

list_files_optical_flow.remove('.ipynb_checkpoints')

complete_final_list = []
for n in range(0, len(list_files_optical_flow)):
    if list_files_optical_flow[n].split('.')[0]+'.csv' in list_files_lstm_data:
        complete_final_list.append(list_files_optical_flow[n].split('.')[0])

len(complete_final_list)

num_frames = 40
frame_height = 200
frame_width = 200

dataset = CustomVideoDataset(video_folders_list=complete_final_list, num_frames=num_frames, frame_height=frame_height, frame_width=frame_width)

validation_split = 0.2
shuffle_dataset = True

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset:
    np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes= 512):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 512)


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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# ResNet3D Model
class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=512):
        super(ResNet3D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return x

def resnet3d():
    return ResNet3D(ResidualBlock, [2, 2, 2, 2])

# Define two ResNet3D models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1_Optical_Flow= resnet3d().to(device)
model_Lstm = LSTMModel(input_size, hidden_size, num_layers, num_classes)

class Attention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(Attention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = self.layer_norm(attn_output + x)
        return attn_output.mean(dim=1)  # Aggregate across the sequence dimension

class CombinedModelWithAttention(nn.Module):
    def __init__(self, model1, model2, feature_dim=512, num_heads=8, num_classes=11):
        super(CombinedModelWithAttention, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.attention = Attention(feature_dim * 2, num_heads)  # feature_dim * 2 because of concatenation
        self.fc_combined = nn.Linear(feature_dim * 2, num_classes)  # Assuming the output of each model is a 512-dimensional feature vector

    def forward(self, x1, x2):
        features1 = self.model1(x1)  # [batch_size, 512]
        features2 = self.model2(x2)  # [batch_size, 512]
        combined_features = torch.cat((features1, features2), dim=1) # [batch_size, 1024]
        combined_features = combined_features.unsqueeze(1)  # [batch_size, 1, 1024]
        attended_features = self.attention(combined_features)  # [batch_size, 1024]
        out = self.fc_combined(attended_features)  # [batch_size, num_classes]
        return out

combined_model_with_attention = CombinedModelWithAttention(model1_Optical_Flow, model_Lstm).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(combined_model_with_attention.parameters(), lr=0.0001)

num_epochs = 100

for epoch in range(num_epochs):
    combined_model_with_attention.train()
    running_loss = 0.0
    for videos1, videos2, labels in train_loader:
        videos1, videos2, labels = videos1.to(device), videos2.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = combined_model_with_attention(videos1, videos2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    combined_model_with_attention.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for videos1, videos2, labels in val_loader:
            videos1, videos2, labels = videos1.to(device), videos2.to(device), labels.to(device)
            outputs = combined_model_with_attention(videos1, videos2)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total


    print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, "
          f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

import torch
from sklearn.metrics import classification_report

# Assuming the following variables are already defined:
# combined_model_with_attention, val_loader, device, action_to_label

# Ensure the model is in evaluation mode
combined_model_with_attention.eval()

all_labels = []
all_predictions = []

# No need to compute gradients for evaluation
with torch.no_grad():
    for videos1, videos2, labels in val_loader:
        videos1, videos2, labels = videos1.to(device), videos2.to(device), labels.to(device)
        outputs = combined_model_with_attention(videos1, videos2)
        _, predicted = torch.max(outputs.data, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# Generate the classification report
report = classification_report(all_labels, all_predictions, target_names=['processed_Arm', 'processed_bs', 'processed_ce', 'processed_dr',
    'processed_fg', 'processed_mfs', 'processed_ms',
    'processed_sq', 'processed_tw', 'processed_sac', 'processed_tr'])
print(report)

