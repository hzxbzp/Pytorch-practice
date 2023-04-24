import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.utils.data as Data
import torch.optim as optim
from matplotlib import pyplot as plt
import random
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y


#create data examples
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

# create dataset and split it into batches
batch_size = 10
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

#build model, loss, and optimizer
net = LinearNet(num_inputs)
init.normal_(net.linear.weight, mean=0,std=0.01)
init.constant_(net.linear.bias, val=0)

loss = nn.MSELoss()

optimizer = optim.SGD(net.parameters(), lr=0.03)


#train model
num_epochs = 3

for epoch in range(num_epochs):
    for x, y in data_iter:
        output = net(x)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch: %d, loss: %f' % (epoch, l.item()))

print(true_w, net.linear.weight)
print(true_b, net.linear.bias)