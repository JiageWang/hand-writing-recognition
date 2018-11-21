import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class HWDB(object):
    def __init__(self, path='./data'):
        # 预处理过程
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Lambda(lambda x: Image.fromarray(255 - np.array(x))),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
        ])

        #
        traindir = os.path.join(path, 'train')
        testdir = os.path.join(path, 'test')

        self.trainset = datasets.ImageFolder(traindir, transform)
        self.testset = datasets.ImageFolder(testdir, transform)
        self.train_size = len(self.trainset)
        self.test_size = len(self.testset)

    def get_sample(self, index=0):
        sample = self.trainset[index]
        sample_img, sample_label = sample
        print(sample_img.size())
        return sample_img, sample_label

    def get_loader(self, batch_size=100):
        train_loader = torch.utils.data.DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            self.testset, batch_size=batch_size, shuffle=True)
        return train_loader, test_loader


if __name__ == '__main__':
    dataset = HWDB()
    for i in [1, 10, 2000, 6000, 1000]:
        img, label = dataset.get_sample(i)
        img = img[0]
        plt.imshow(img, cmap='gray')
        plt.show()

    train_loader, test_loader = dataset.get_loader()
    for  (img, label) in train_loader:
        print(img)
        print(label)






