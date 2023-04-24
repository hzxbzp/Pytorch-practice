import numpy as np
import torch
from matplotlib import pyplot as plt
import random



def data_loader(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)


def linreg(x, w, b):
    return torch.mm(x,w) + b


def squared_loss(yhat, y):
    return 0.5 * (yhat - y.view(yhat.size())) ** 2


def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


# create data examples
num_inputs = 2
num_examples = 300
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
true_w = [2, -3.4]
true_b = 4.2
labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)

#initialize model parameters
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


#train model
lr = 0.03
num_epochs = 3
batch_size = 10
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for x, y in data_loader(batch_size, features, labels):
        l = loss(net(x, w, b), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)

        w.grad.data.zero_()
        b.grad.data.zero_()

    train_l = loss(net(features, w, b), labels)
    print('epoch: %d,  loss: %f' % (epoch, train_l.mean().item()))

print(true_w, '\n', w)
print(true_b, '\n', b)


# regression
w.requires_grad_(requires_grad=False)
b.requires_grad_(requires_grad=False)
yhats = net(features, w, b)


# draw data examples
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(features[:,0].numpy(), features[:,0].numpy(), labels.numpy(), c='g')
ax.scatter(features[:,0].numpy(), features[:,0].numpy(), yhats.numpy(), c='r')
ax.set(xlabel='x1',ylabel='x2',zlabel='y')
plt.show()

