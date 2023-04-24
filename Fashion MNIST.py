import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import init
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')

def show_fashion_mnist(images, labels):
    use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        y = self.linear(x.view(x.shape[0], -1))
        return y

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        yhat = net(X).argmax(dim=1)
        acc_sum += (yhat == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def train(train_iter, test_iter, num_epochs, batch_size, net, optimizer, loss):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for x, y in train_iter:
            y_hat = net(x)
            l = loss(y_hat, y).sum()

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum +=l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch, train_l_sum / n, train_acc_sum / n, test_acc))



# load datasets
mnist_train = torchvision.datasets.FashionMNIST(root='D:/Personal Document/PhD/Code/pytorch/Datasets', download=True,train=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='D:/Personal Document/PhD/Code/pytorch/Datasets', download=True,train=False, transform=transforms.ToTensor())

batch_size =  256
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

#build model, loss, optimizer
num_inputs = 784
num_outputs = 10
num_epochs = 5

net = LinearNet(num_inputs, num_outputs)
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# train the model
train(train_iter, test_iter, num_epochs, batch_size, net, optimizer, loss)

# predict
X, y = next(iter(test_iter))

true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

show_fashion_mnist(X[0:9], titles[0:9])




