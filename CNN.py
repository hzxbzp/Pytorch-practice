import torch, torch.nn as nn


# 单输入通道卷积运算（实际为互相关运算 cross correlation）
def corr2d(x, k, stride):
    h, w = k.shape[0], k.shape[1]
    y = torch.zeros((int((x.shape[0] - h) / stride + 1), int((x.shape[1] - w) / stride + 1)))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = (x[i:i + h, j:j + w] * k).sum()
    return y


class Conv2D(nn.Module):
    def __init__(self, kernel_size, stride):
        super(Conv2D, self).__init__()
        self.stride = stride
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight, self.stride) + self.bias


# 使用物体边缘检测中的输入数据X和输出数据Y来学习我们构造的核数组K
# X = torch.ones(6, 8)
# X[:, 2:6] = 0
# K = torch.tensor([[1, -1]])
# y = corr2d(X, K,stride=1)
#
# conv2d = Conv2D(kernel_size=(1, 2), stride=1)
#
# step = 50
# lr = 0.01
#
# for i in range(step):
#     yhat = conv2d(X)
#     l = ((yhat - y) ** 2).sum()
#     l.backward()
#
#     # 梯度下降
#     conv2d.weight.data -= lr * conv2d.weight.grad
#     conv2d.bias.data -= lr * conv2d.bias.grad
#
#     # 梯度清0
#     conv2d.weight.grad.fill_(0)
#     conv2d.bias.grad.fill_(0)
#
#     if (i + 1) % 5 == 0:
#         print('Step %d, loss %.3f' % (i + 1, l.item()))
#
# print("weight: ", conv2d.weight.data)
# print("bias: ", conv2d.bias.data)

# 多输入通道卷积运算
def corr2d_multi_in(x, k, stride):
    res = corr2d(x[0, :, :], k[0, :, :], stride=stride)
    for i in range(1, x.shape[0]):
        res += corr2d(x[i, :, :], k[i, :, :], stride=stride)
    return res


# 多输出通道卷积运算
def corr2d_multi_in_out(X, K, stride):
    return torch.stack([corr2d_multi_in(X, k, stride) for k in K])


# X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
#                   [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
# K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])
# K = torch.stack([K, K + 1, K + 2])
# print(corr2d_multi_in_out(X, K, stride=1))


# 2*2池化    池化层的一个主要作用是缓解卷积层对位置的过度敏感性
def pool2d(X, pool_size, mode='max'):
    X = X.float()
    pw, ph = pool_size
    Y = torch.zeros(X.shape[0] - ph + 1, X.shape[1] - pw + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + ph, j: j + pw].max()
            elif mode =='avg':
                Y[i, j] = X[i: i + ph, j: j + pw].mean()
    return Y


X1 = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
Y1 = pool2d(X1, (2, 2), 'avg')

# nn.MaxPool2d
X2 = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))
maxpool2d = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))
Y2 = maxpool2d(X2)


X2 = torch.cat((X2, X2 + 1), dim=1)
maxpool2d2 = nn.MaxPool2d(3, padding=1, stride=2)
Y3 = maxpool2d2(X2)
print(Y3)