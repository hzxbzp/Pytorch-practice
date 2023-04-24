import time
import torch, torch.nn as nn, torch.optim as optim
from utils import FlattenLayer, load_fashion_mnist, train


def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*blk)


def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        net.add_module('vgg_block_' + str(i + 1), vgg_block(num_convs, in_channels, out_channels))
    net.add_module('fc', nn.Sequential(FlattenLayer(),
                                       nn.Linear(fc_features, fc_hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units, fc_hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units, 10)
                                       ))
    return net


# conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
# # 经过5个vgg_block后，高宽会减半5次 224 / 32 = 7
# fc_features = 512 * 7 * 7
# fc_hidden_units = 4096
# net = vgg(conv_arch, fc_features, fc_hidden_units)
#
# X = torch.rand(1, 1, 224, 224)
# for name, blk in net.named_children():
#     X = blk(X)
#     print(name, 'output shape: ', X.shape)

# 设置超参数
ratio = 8
small_conv_arch = ((1, 1, 64//ratio), (1, 64//ratio, 128//ratio), (2, 128//ratio, 256//ratio),
                   (2, 256//ratio, 512//ratio), (2, 512//ratio, 512//ratio))
fc_features = 512 * 7 * 7
fc_hidden_units = 4096
batch_size = 64
lr, num_epochs = 0.001, 5

# 加载数据
train_iter, test_iter = load_fashion_mnist(batch_size, resize=224)

# 实例化
net = vgg(small_conv_arch, fc_features // ratio, fc_hidden_units // ratio)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = optim.Adam(net.parameters(), lr=lr)
train(net, train_iter, test_iter, optimizer, device, num_epochs)
