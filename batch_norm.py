import time
import torch
from torch import nn, optim
import torch.nn.functional as F

def batch_norm(is_training, x, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not is_training:
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(x.shape) in (2, 4)
        if len(x.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = x.mean(dim=0)
            var = ((x - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。这里我们需要保持
            # X的形状以便后面可以做广播运算
            mean = x.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((x - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)

        x_hat = (x - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    y = gamma * x_hat + beta
    return y, moving_mean, moving_var


class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成0和1
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 不参与求梯度和迭代的变量，全在内存上初始化成0
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, x):
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在显存上
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.var = self.moving_var.to(x.device)
        # 保存更新过的moving_mean和moving_var, Module实例的traning属性默认为true, 调用.eval()后设成false
        y, self.moving_mean, self.moving_var = batch_norm(self.training,
                                                          x, self.gamma, self.beta, self.moving_mean,
                                                          self.moving_var, eps=1e-5, momentum=0.9)
        return y