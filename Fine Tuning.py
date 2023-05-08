import torch
from torch import nn, optim
from  torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import os
import matplotlib.pyplot as plt
from utils import show_images, train


# 指定RGB三个通道的均值和方差来将图像通道归一化
train_augs = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

test_augs = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

# 加载数据
data_dir = './Datasets'
train_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/train'), transform=train_augs)
test_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/test'), transform=test_augs)

# 修改输出层
pretrained_net = models.resnet18(pretrained=True)
pretrained_net.fc = nn.Linear(512, 2)

# 对输出层和其他层设置不同的学习率。 其他层学习率小，对参数进行微调
output_params = list(map(id, pretrained_net.fc.parameters()))
print(output_params)
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())
lr = 0.01
optimizer = optim.SGD([{'params': feature_params},
                       {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
                      lr=lr, weight_decay=0.001)


def train_fine_tuning(net, optimizer, batch_size=128, num_epochs=5):
    train_iter = DataLoader(train_imgs, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(train_imgs, batch_size=batch_size)
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train(net, train_iter, test_iter, optimizer, device, num_epochs)


train_fine_tuning(pretrained_net, optimizer)
