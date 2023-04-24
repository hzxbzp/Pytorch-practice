import torch, torch.nn as nn
import torchvision, torchvision.transforms as transforms
import time
import torch.nn.functional as F

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


def load_fashion_mnist(batch_size, resize=None):
    trans = []
    if resize:
        trans.append((torchvision.transforms.Resize(size=resize)))
    trans.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(root='D:/Personal Document/PhD/Code/pytorch/Datasets',
                                                    download=True, train=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root='D:/Personal Document/PhD/Code/pytorch/Datasets',
                                                   download=True, train=False, transform=transform)

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

    return train_iter, test_iter


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list((net.parameters()))[0].device

    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                acc_sum += (net(x.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()
            else:  # 自定义模型
                if 'is_training' in net.__code__.co_varnames:  # 如果有is_training这个参数
                    acc_sum += (net(x, is_training=False).argmax(dim=1) == y).float().sum().cpu().item()
                else:
                    acc_sum += (net(x).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def train(net, train_iter, test_iter, optimizer, device, num_epochs):
    net = net.to(device)
    print('training on', device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for x, y in train_iter:
            x = x.to(device)
            y = y.to(device)
            yhat = net(x)
            l = loss(yhat, y)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.cpu().item()
            train_acc_sum += (yhat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train_acc %.3f, test_acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
