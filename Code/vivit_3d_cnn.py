!pip install timm
!pip install tensorboardX
!pip install einops

import os
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import torch.utils.checkpoint as checkpoint

class CustomVideoDataset(Dataset):
    def __init__(self, video_folders_list, num_frames, frame_height, frame_width):
        self.video_files_1 =  video_folders_list

        self.num_frames = num_frames
        self.frame_height = frame_height
        self.frame_width = frame_width

    def __len__(self):
        return len(self.video_files_1)

    def __getitem__(self, idx):
        optical_flow_video = 'Complete_Optical_Flow_20_Frames_ViViT_Star' +  '/' + self.video_files_1[idx]
        romp_videos = 'Complete_Romp_Video_All_20_Frames_For_ViVit' + '/' + self.video_files_1[idx]

        video_path_1 = optical_flow_video
        video_path_2 = romp_videos

        frames_1 = self._load_frames(video_path_1)
        frames_2 = self._load_frames(video_path_2)

        video_action = '_'.join(self.video_files_1[idx].split('_')[:2])
        label = 0
        action_to_label = {
            'processed_Arm': 0, 'processed_bs': 1, 'processed_ce': 2, 'processed_dr': 3,
            'processed_fg': 4, 'processed_mfs': 5, 'processed_ms': 6,
            'processed_sq': 7, 'processed_tw': 8, 'processed_sac': 9, 'processed_tr': 10
        }

        label = action_to_label.get(video_action, label)
        label = torch.tensor(label, dtype=torch.long)

        return frames_1, frames_2, label

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

        while len(frames) < self.num_frames:
            frames.append(np.zeros((self.frame_height, self.frame_width), dtype=np.float32))  # Gray frame

        video_tensor = torch.tensor(np.stack(frames, axis=0)).unsqueeze(1).float() / 255  # Convert to torch tensor and normalize to [0,1]
        video_tensor = video_tensor.permute(1, 0, 2, 3)  # Reorder dimensions to [1, 16, 224, 224]

        return video_tensor

video_folder_optical_flow =  'Complete_Optical_Flow_20_Frames_ViViT_Star_New'
video_romp_video = 'Complete_Romp_Video_All_20_Frames_For_ViVit'

list_files_optical_flow = os.listdir(video_folder_optical_flow)
list_files_romp_video = os.listdir(video_romp_video)
final_video_files =   set(list_files_romp_video) - set(list_files_optical_flow)

complete_final_list = []
for n in range(0, len(list_files_romp_video)):
    if list_files_romp_video[n] in list_files_optical_flow:
        complete_final_list.append(list_files_romp_video[n])

len(complete_final_list)



num_frames = 20
frame_height = 100
frame_width = 100

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

class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=100, patch_size=10, in_chans=1, embed_dim=200, num_frames=20, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim,
                            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
                            stride=(self.tubelet_size, patch_size[0], patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)

class PretrainVisionTransformerEncoder(nn.Module):
    def __init__(self, img_size=100, patch_size=10, in_chans=1, num_classes=0, embed_dim=200, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2, use_checkpoint=False,
                 use_learnable_pos_emb=False, num_frames=20):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            num_frames=num_frames, tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint

        self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        B, _, C = x.shape

        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
                x_vis = x
        else:
            for blk in self.blocks:
                x = blk(x)
                x_vis = x

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x):
        x = self.forward_features(x)
        #print(x.shape)
        #x = self.head(x)
        x = x.mean(dim=1)
        #x = self.mlp_head(x)
        return x

img_size = 100
patch_size = 10
in_chans = 1
num_classes = 11
embed_dim = 400
depth = 12
num_heads = 12
mlp_ratio = 4.0
qkv_bias = False
qk_scale = None
drop_rate = 0.0
attn_drop_rate = 0.0
drop_path_rate = 0.0
norm_layer = nn.LayerNorm
init_values = None
tubelet_size = 2
use_checkpoint = False
use_learnable_pos_emb = False
num_frames = 20

model_Optical_Flow  = PretrainVisionTransformerEncoder(
    img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
    embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
    qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
    norm_layer=norm_layer, init_values=init_values, tubelet_size=tubelet_size, use_checkpoint=use_checkpoint,
    use_learnable_pos_emb=use_learnable_pos_emb, num_frames=num_frames)

model_Romp_Video = PretrainVisionTransformerEncoder(
    img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
    embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
    qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
    norm_layer=norm_layer, init_values=init_values, tubelet_size=tubelet_size, use_checkpoint=use_checkpoint,
    use_learnable_pos_emb=use_learnable_pos_emb, num_frames=num_frames)

class MultiModalModel(nn.Module):
    def __init__(self, img_size=100, patch_size=10, in_chans=1, num_classes=11, embed_dim=200, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2, use_checkpoint=False, num_frames=20):
        super(MultiModalModel, self).__init__()
        self.optical_flow_model = model_Optical_Flow
        self.romp_model = model_Romp_Video
        self.attention = Attention(dim=embed_dim*2, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate, proj_drop=drop_rate)
        self.fc = nn.Linear(embed_dim*2, num_classes)

    def forward(self, optical_flow, romp):
        optical_flow_features = self.optical_flow_model(optical_flow)
        romp_features = self.romp_model(romp)
        combined_features = torch.cat((optical_flow_features, romp_features), dim=-1)
        attended_features = self.attention(combined_features.unsqueeze(1)).squeeze(1)
        logits = self.fc(attended_features)
        return logits

model = MultiModalModel(
    img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
    embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
    qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
    norm_layer=norm_layer, init_values=init_values, tubelet_size=tubelet_size, use_checkpoint=use_checkpoint,
    num_frames=num_frames
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#num_classes = 11
num_epochs = 70
learning_rate = 1e-4

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for videos1, videos2, labels in train_loader:
        videos1, videos2, labels = videos1.to(device), videos2.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(videos1, videos2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for videos1, videos2, labels in val_loader:
            videos1, videos2, labels = videos1.to(device), videos2.to(device), labels.to(device)
            outputs = model(videos1, videos2)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    #if epoch %  == 0:
    torch.save(model.state_dict(), 'ViViT_Combined_Weights_Optical_Flow_Romp_Videos/' + f'combined_model_epoch_{epoch}.pth')

    print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, "
          f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

pip install scikit-learn

import torch
from sklearn.metrics import classification_report

model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():
    for videos1, videos2, labels in val_loader:
        videos1, videos2, labels = videos1.to(device), videos2.to(device), labels.to(device)
        outputs = model(videos1, videos2)
        _, predicted = torch.max(outputs.data, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

report = classification_report(all_labels, all_predictions, target_names=['processed_Arm', 'processed_bs', 'processed_ce', 'processed_dr',
    'processed_fg', 'processed_mfs', 'processed_ms',
    'processed_sq', 'processed_tw', 'processed_sac', 'processed_tr'])
print(report)

