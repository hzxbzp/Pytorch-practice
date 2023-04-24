import torch, torch.nn as nn
import torch.nn.init as init

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(784, 256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)


    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

class FancyMLP(nn.Module):
    def __init__(self):
        super(FancyMLP, self).__init__()
        self.rand_weight = torch.rand((20,20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)
        x = self.linear(x)
        return x.sum()

class NestMLP(nn.Module):
    def __init__(self):
        super(NestMLP, self).__init__()
        self.net = nn.Sequential(nn.Linear(40, 30), nn.ReLU())

    def forward(self, x):
        return self.net(x)


x = torch.rand(2, 40)
net = nn.Sequential(NestMLP(), nn.Linear(30, 20), FancyMLP())

# look at parameters of the net
# for name, param in net[0].named_parameters():
#     print(name, param.size())
#
# weight_0 = list(net[0].parameters())[0]
# print(weight_0.data)
# print(weight_0.grad)

# initialize parameters of the net by using nn.init
# for name, param in net.named_parameters():
#     if 'weight' in name:
#         init.normal_(param, mean=0, std=0.01)
#         print(name, param.data)
#
#     if 'bias' in name:
#         init.constant_(param, val=0)
#         print(name, param.data)

# define your own initialization method
# def init_weight_(tensor):
#     with torch.no_grad():
#         tensor.uniform_(-10, 10)
#         tensor *= (tensor.abs() >= 5).float()
#
# for name, param in net.named_parameters():
#     if 'weight' in name:
#         init_weight_(param)
#         print(name, param.data)



class MyDense(nn.Module):
    def __init__(self):
        super(MyDense, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))

    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x

net1 = MyDense()

# 保存和加载模型参数  save
# torch.save(net.state_dict(), 'path') # 推荐的文件后缀名是pt或pth
# # load
# net = nn.Sequential(NestMLP(), nn.Linear(30, 20), FancyMLP())
# net.load_state_dict(torch.load('path'))

# 保存和加载整个模型
# torch.save(model, PATH)
# model = torch.load(PATH)

# 将变量存放到GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
a = torch.tensor([1,2,3], device=device)
# or a = torch.tensor([1,2,3]).to(device)
#    a = a.cuda()

# 将model存放到GPU   模型和输入的tensor需要在同一设备上
net.to(device)
net.cuda()