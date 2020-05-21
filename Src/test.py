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

            nn.Sequential(
                # Pointwise Convolution
                nn.Conv2d(bandwidth[0], bandwidth[0], 1),
                # Depthwise Convolution
                nn.Conv2d(bandwidth[0], bandwidth[0], 3, 1, 1, groups=bandwidth[0]),
                # Batch Normalization
                nn.BatchNorm2d(bandwidth[0]),
                # ReLU6 是限制Neuron最小只會到0，最大只會到6。 MobileNet系列都是使用ReLU6。
                # 使用ReLU6的原因是因為如果數字太大，會不好壓到float16 / or further qunatization，因此才給個限制。
                nn.ReLU6(),
                # Pointwise Convolution
                nn.Conv2d(bandwidth[0], bandwidth[1], 1),
                # 過完Pointwise Convolution不需要再做ReLU，經驗上Pointwise + ReLU效果都會變差。
                nn.MaxPool2d(2, 2, 0),
                # 每過完一個Block就Down Sampling
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[1], bandwidth[1], 1),
                nn.Conv2d(bandwidth[1], bandwidth[1], 3, 1, 1, groups=bandwidth[1]),
                nn.BatchNorm2d(bandwidth[1]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[1], bandwidth[2], 1),
                nn.MaxPool2d(2, 2, 0),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[2], bandwidth[2], 1),
                nn.Conv2d(bandwidth[2], bandwidth[2], 3, 1, 1, groups=bandwidth[2]),
                nn.BatchNorm2d(bandwidth[2]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[2], bandwidth[3], 1),
                nn.MaxPool2d(2, 2, 0),
            ),

            # 到這邊為止因為圖片已經被Down Sample很多次了，所以就不做MaxPool
            nn.Sequential(
                nn.Conv2d(bandwidth[3], bandwidth[3], 1, groups=bandwidth[3]//4),
                nn.Conv2d(bandwidth[3], bandwidth[3], 3, 1, 1, groups=bandwidth[3]),
                nn.BatchNorm2d(bandwidth[3]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[3], bandwidth[4], 1, groups=bandwidth[3]//4),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[4], bandwidth[4], 1, groups=bandwidth[4]//16),
                nn.Conv2d(bandwidth[4], bandwidth[4], 3, 1, 1, groups=bandwidth[4]),
                nn.BatchNorm2d(bandwidth[4]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[4], bandwidth[5], 1, groups=bandwidth[4]//16),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[5], bandwidth[5], 1, groups=bandwidth[4]//16),
                nn.Conv2d(bandwidth[5], bandwidth[5], 3, 1, 1, groups=bandwidth[5]),
                nn.BatchNorm2d(bandwidth[5]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[5], bandwidth[6], 1, groups=bandwidth[4]//16),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[6], bandwidth[6], 1, groups=bandwidth[4]//32),
                nn.Conv2d(bandwidth[6], bandwidth[6], 3, 1, 1, groups=bandwidth[6]),
                nn.BatchNorm2d(bandwidth[6]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[6], bandwidth[7], 1, groups=bandwidth[4]//32),
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


def readfile(path, label):
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

testTransform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

print('Load data...')
workspace_dir = sys.argv[1]
test_x = readfile(os.path.join(workspace_dir, "testing"), False)
test_set = MyDataset(test_x, transform=testTransform)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

student_net = StudentNet(base=16).cuda()
student_net.load_state_dict(torch.load('./student_model_16.bin'))

student_net.eval()
prediction = []

with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = student_net(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

with open(sys.argv[2], mode='w') as f:
    f.write('Id,label\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))
print('---End Test---')