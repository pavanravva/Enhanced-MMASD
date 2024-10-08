import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
import numpy as np
import torch.nn.functional as F

# Custom Video Dataset
class CustomVideoDataset(Dataset):
    def __init__(self, video_folder, num_frames, frame_height, frame_width):
        self.video_folder = video_folder
        self.video_files = []

        for folder in video_folder:
            folder_path = os.path.join(folder)
            self.video_files.extend([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mp4')])
        self.num_frames = num_frames
        self.frame_height = frame_height
        self.frame_width = frame_width

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
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

        video_action = "_".join(video_path.split("/")[-2].split('_')[:2])
        label = 0
        action_to_label = {
            'Arm_Swing': 0, 'Body_Swing': 1, 'Chest_Expansion': 2, 'Drumming_Pose': 3,
            'Frog_Pose': 4, 'Marcas_Forward': 5, 'Marcas_Shaking': 6,
            'Squat_Pose': 7, 'Twist_Pose': 8, 'Sing_Clap': 9, 'Tree_Pose': 10
        }


        label = action_to_label.get(video_action, label)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)

        return video_tensor, label

# Residual Block
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
    def __init__(self, block, layers, num_classes=11):
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
        action_output = self.fc(x)
        return action_output

def resnet3d():
    return ResNet3D(ResidualBlock, [2, 2, 2, 2])

video_folder = [


    'Folder_Data',


]


num_frames = 40
frame_height = 200
frame_width = 200

dataset = CustomVideoDataset(video_folder=video_folder, num_frames=num_frames, frame_height=frame_height, frame_width=frame_width)

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

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet3d().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 100



for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for videos, labels in train_loader:
        videos, labels = videos.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    if epoch % 10 == 0:
        torch.save(model.state_dict(), 'CNN_Weights/' + f'model_epoch_{epoch}_new.pth')



    print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, "
          f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

from sklearn.metrics import f1_score
from sklearn.metrics import f1_score, classification_report

# Evaluate on the validation set
model.eval()
all_labels = []
all_preds = []
with torch.no_grad():
    for videos, labels in val_loader:
        videos, labels = videos.to(device), labels.to(device)
        outputs = model(videos)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

f1 = f1_score(all_labels, all_preds, average='weighted')  # Overall F1-score (weighted)
class_report = classification_report(all_labels, all_preds, target_names=[
    'Arm_Swing', 'Body_Swing', 'Chest_Expansion', 'Drumming_Pose', 'Frog_Pose',
    'Marcas_Forward', 'Marcas_Shaking', 'Squat_Pose', 'Twist_Pose', 'Sing_Clap', 'Tree_Pose'
])

print(f"Final F1 Score (weighted): {f1:.4f}")
print("Class-wise F1-scores:")
print(class_report)

