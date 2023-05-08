import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
from utils import set_figsize, show_images
from matplotlib import pyplot as plt


# def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
#     y = [aug(img) for _ in range(num_rows * num_cols)]
#     show_images(y, num_rows, num_cols, scale)
#
#
# set_figsize()
# img = Image.open('./Datasets/personal.jpg')

# apply(img, torchvision.transforms.RandomHorizontalFlip())
# apply(img, torchvision.transforms.RandomVerticalFlip())

# shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
# apply(img, shape_aug)

# color_aug = torchvision.transforms.ColorJitter(
#     brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
# apply(img, color_aug)

# augs = torchvision.transforms.Compose([
#     torchvision.transforms.RandomHorizontalFlip(),
#     torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
#     torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
# ])
# apply(img, augs)
# plt.show()


all_images = torchvision.datasets.CIFAR10(train=True, root='./Datasets/CIFAR', download=True)
# all_imges的每一个元素都是(image, label)
show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)
plt.show()

flip_aug = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()
])

no_aug = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

def load_cifar10(is_train, augs, batch_size, root="~/Datasets/CIFAR"):
    dataset = torchvision.datasets.CIFAR10(root=root, train=is_train, transform=augs, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=0)