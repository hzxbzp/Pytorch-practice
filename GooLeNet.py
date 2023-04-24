import time
import torch, torch.nn as nn, torch.optim as optim
from utils import GlobalAvgPool2d, FlattenLayer, load_fashion_mnist, train


class Inception(nn.Module):
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.p1 = nn.Sequential(
            nn.Conv2d(in_c, c1, kernel_size=1),
            nn.ReLU()
        )
        self.p2 = nn.Sequential(
            nn.Conv2d(in_c, c2[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(in_c, c3[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_c, c4, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        return torch.cat((p1, p2, p3, p4), dim=1)


b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                   )

b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                   )

b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                   )

b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                   )

b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   GlobalAvgPool2d()
                   )

net = nn.Sequential(b1, b2, b3, b4, b5,
                    FlattenLayer(),
                    nn.Linear(1024, 10)
                    )

batch_size = 128
train_iter, test_iter = load_fashion_mnist(batch_size, resize=96)

lr, num_epochs = 0.001, 5
optimizer = optim.Adam(net.parameters(), lr=lr)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train(net, train_iter, test_iter, optimizer, device, num_epochs)