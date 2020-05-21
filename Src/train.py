import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torch
import os
import sys
import re
from glob import glob
from PIL import Image
from cv2 import cv2
import numpy as np
import torchvision.transforms as transforms

class StudentNet(nn.Module):
    def __init__(self, base=16, width_mult=1):
        '''
          Args:
            base: 這個model一開始的ch數量，每過一層都會*2，直到base*16為止。
            width_mult: 為了之後的Network Pruning使用，在base*8 chs的Layer上會 * width_mult代表剪枝後的ch數量。        
        '''
        super(StudentNet, self).__init__()
        multiplier = [1, 2, 4, 8, 16, 16, 16, 16]

        # bandwidth: 每一層Layer所使用的ch數量
        bandwidth = [ base * m for m in multiplier]

        # 我們只Pruning第三層以後的Layer
        for i in range(3, 7):
            bandwidth[i] = int(bandwidth[i] * width_mult)

        self.cnn = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, bandwidth[0], 3, 1, 1),
                nn.BatchNorm2d(bandwidth[0]),
                nn.ReLU6(),
                nn.MaxPool2d(2, 2, 0),
            ),

            nn.Sequential( # 16*9 + 16*32 + 32*9 + 32*32
                nn.Conv2d(bandwidth[0], bandwidth[0], 3, 1, 1, groups=bandwidth[0]),
                nn.BatchNorm2d(bandwidth[0]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[0], bandwidth[1], 1),
                # Depthwise Convolution
                nn.Conv2d(bandwidth[1], bandwidth[1], 3, 1, 1, groups=bandwidth[0]),
                # Batch Normalization
                nn.BatchNorm2d(bandwidth[1]),
                # ReLU6 是限制Neuron最小只會到0，最大只會到6。 MobileNet系列都是使用ReLU6。
                # 使用ReLU6的原因是因為如果數字太大，會不好壓到float16 / or further qunatization，因此才給個限制。
                nn.ReLU6(),
                # Pointwise Convolution
                nn.Conv2d(bandwidth[1], bandwidth[1], 1),
                # 過完Pointwise Convolution不需要再做ReLU，經驗上Pointwise + ReLU效果都會變差。
                nn.MaxPool2d(2, 2, 0),
                # 每過完一個Block就Down Sampling
            ),

            nn.Sequential( # 32*9 + 32*64 + 64*9 + 64*64
                nn.Conv2d(bandwidth[1], bandwidth[1], 3, 1, 1, groups=bandwidth[1]),
                nn.BatchNorm2d(bandwidth[1]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[1], bandwidth[2], 1),
                nn.Conv2d(bandwidth[2], bandwidth[2], 3, 1, 1, groups=bandwidth[1]),
                nn.BatchNorm2d(bandwidth[2]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[2], bandwidth[2], 1),
                nn.MaxPool2d(2, 2, 0),
            ),

            nn.Sequential( # 64*9 + 64*128 + 128*9 + 128*128
                nn.Conv2d(bandwidth[2], bandwidth[2], 3, 1, 1, groups=bandwidth[2]),
                nn.BatchNorm2d(bandwidth[2]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[2], bandwidth[3], 1),
                nn.Conv2d(bandwidth[3], bandwidth[3], 3, 1, 1, groups=bandwidth[2]),
                nn.BatchNorm2d(bandwidth[3]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[3], bandwidth[3], 1),
                nn.MaxPool2d(2, 2, 0),
            ),

            # 到這邊為止因為圖片已經被Down Sample很多次了，所以就不做MaxPool
            nn.Sequential( # 128*9 + (128*256) 8*16*16
                nn.Conv2d(bandwidth[3], bandwidth[3], 3, 1, 1, groups=bandwidth[3]),
                nn.BatchNorm2d(bandwidth[3]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[3], bandwidth[4], 1, groups=bandwidth[3]//8),
            ),

            nn.Sequential( # 256*9 + 256*256 // 8*8*32 // 16*16*16
                nn.Conv2d(bandwidth[4], bandwidth[4], 3, 1, 1, groups=bandwidth[4]),
                nn.BatchNorm2d(bandwidth[4]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[4], bandwidth[5], 1, groups=bandwidth[4]//16),
            ),

            nn.Sequential( # 256*9 + 256*256 // 8*8*32 // 16*16*16
                nn.Conv2d(bandwidth[5], bandwidth[5], 3, 1, 1, groups=bandwidth[5]),
                nn.BatchNorm2d(bandwidth[5]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[5], bandwidth[6], 1, groups=bandwidth[4]//16),
            ),

            nn.Sequential( # 256*9 + 256*256 // 8*8*32 // 16*16*16
                nn.Conv2d(bandwidth[6], bandwidth[6], 3, 1, 1, groups=bandwidth[6]),
                nn.BatchNorm2d(bandwidth[6]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[6], bandwidth[7], 1, groups=bandwidth[4]//16),
            ),

            # 這邊我們採用Global Average Pooling。
            # 如果輸入圖片大小不一樣的話，就會因為Global Average Pooling壓成一樣的形狀，這樣子接下來做FC就不會對不起來。
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            # 這邊我們直接Project到11維輸出答案。
            nn.Linear(bandwidth[7], 11),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):
    # 一般的Cross Entropy
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    # 讓logits的log_softmax對目標機率(teacher的logits/T後softmax)做KL Divergence。
    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    return hard_loss + soft_loss

def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 512, 512, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        ratio = 512/max(h, w)
        nh, nw = int(h*ratio), int(w*ratio)
        img = cv2.resize(img, (nw, nh))
        t = (512-nh)//2
        l = (512-nw)//2
        x[i, :, :] = cv2.resize(cv2.copyMakeBorder(img, t,512-nh-t, l,512-nw-l, cv2.BORDER_REFLECT), (512, 512))
        if label:
            y[i] = int(file.split("_")[0])
    if label:
        return x, y
    else:
        return x

class MyDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

trainTransform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(256, pad_if_needed=True, padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])
testTransform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

print('Load data...')
workspace_dir = sys.argv[1]
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
#train_val_x = np.concatenate((train_x, val_x), axis=0)
#train_val_y = np.concatenate((train_y, val_y), axis=0)

#train_val_set = MyDataset(train_val_x, train_val_y, trainTransform)
#train_val_loader = DataLoader(train_val_set, batch_size=32, shuffle=True)

train_set = MyDataset(train_x, train_y, trainTransform)
val_set = MyDataset(val_x, val_y, testTransform)

train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(val_set, batch_size=32, shuffle=False)

print('Load teacher model')
student_net = StudentNet(base=16).cuda()
teacher_net = models.resnet18(pretrained=False, num_classes=11).cuda()

teacher_net.load_state_dict(torch.load(f'./teacher_resnet18.bin'))

adam = optim.Adam(student_net.parameters(), lr=0.01)
sgdm = optim.SGD(student_net.parameters(), lr=0.005, momentum=0.9)

def run_epoch(dataloader, update=True, alpha=0.5, epoch=0):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):

        inputs, hard_labels = batch_data
        inputs = inputs.cuda()
        hard_labels = torch.LongTensor(hard_labels).cuda()
        
        with torch.no_grad():
            soft_labels = teacher_net(inputs)

        if update:
            logits = student_net(inputs)
            if epoch < 300:
                loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, 0.5)
                adam.zero_grad()
                loss.backward()
                adam.step()
            else:
                loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, 0)
                sgdm.zero_grad()
                loss.backward()
                sgdm.step()    
        else:
            with torch.no_grad():
                logits = student_net(inputs)
                loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            
        total_hit += torch.sum(torch.argmax(logits, dim=1) == hard_labels).item()
        total_num += len(inputs)

        total_loss += loss.item() * len(inputs)
    return total_loss / total_num, total_hit / total_num

print('Start training...')
# TeacherNet永遠都是Eval mode.
teacher_net.eval()
now_best_acc = 0
for epoch in range(200):
    student_net.train()
    #train_loss, train_acc = run_epoch(train_val_loader, update=True, epoch=epoch)
    train_loss, train_acc = run_epoch(train_dataloader, update=True, epoch=epoch)
    student_net.eval()
    valid_loss, valid_acc = run_epoch(valid_dataloader, update=False, epoch=epoch)

    # 存下最好的model。
    if valid_acc > now_best_acc:
        now_best_acc = valid_acc
        torch.save(student_net.state_dict(), 'student_model.bin')
    print('epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f}, valid loss: {:6.4f}, acc {:6.4f}'.format(epoch, train_loss, train_acc, valid_loss, valid_acc))
