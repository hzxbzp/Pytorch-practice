import torch, torch.nn as nn
import torch.optim as optim
import torchvision, torchvision.transforms as transforms
import time
from utils import load_fashion_mnist, train


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


batch_size = 256
train_iter, test_iter = load_fashion_mnist(batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = LeNet()

lr, num_epochs = 0.001, 10
optimizer = optim.Adam(net.parameters(), lr=lr)
train(net, train_iter, test_iter, optimizer, device, num_epochs)
