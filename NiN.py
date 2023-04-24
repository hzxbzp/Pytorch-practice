import time
import torch
from torch import nn, optim
from utils import GlobalAvgPool2d, FlattenLayer, load_fashion_mnist, train


def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU()
                        )
    return blk

net = nn.Sequential(
    nin_block(1, 96, 11, 4, 0),
    nn.MaxPool2d(3, 2),
    nin_block(96, 256, 5, 1, 2),
    nn.MaxPool2d(3, 2),
    nin_block(256, 384, 3, 1, 1),
    nn.MaxPool2d(3, 2),
    nn.Dropout(0.5),
    nin_block(384, 10, 3, 1, 1),
    GlobalAvgPool2d(),
    FlattenLayer()
)

batch_size = 128
train_iter, test_iter = load_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.002, 5

optimizer = torch.optim.Adam(net.parameters(), lr=lr)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train(net, train_iter, test_iter, optimizer, device, num_epochs)