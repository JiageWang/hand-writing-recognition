import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt


class HWDB(object):
    def __init__(self, transform, path='./data'):
        # 预处理过程

        traindir = os.path.join(path, 'train')
        testdir = os.path.join(path, 'test')

        self.trainset = datasets.ImageFolder(traindir, transform)
        self.testset = datasets.ImageFolder(testdir, transform)
        self.train_size = len(self.trainset)
        self.test_size = len(self.testset)

    def get_sample(self, index=0):
        sample = self.trainset[index]
        sample_img, sample_label = sample
        return sample_img, sample_label

    def get_loader(self, batch_size=100):
        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(self.testset, batch_size=batch_size, shuffle=True)
        return trainloader, testloader


if __name__ == '__main__':
    transform = transforms.Compose([
        # transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    dataset = HWDB(transform=transform, path=r'C:\Users\Administrator\Desktop\hand-writing-recognition\data')
    print(dataset.train_size)
    print(dataset.test_size)
    for i in [1020, 120, 2000, 6000, 1000]:
        img, label = dataset.get_sample(i)
        img = img[0]
        print(label)
        plt.imshow(img, cmap='gray')
        plt.show()

    train_loader, test_loader = dataset.get_loader()
    print(len(train_loader))
    # for (img, label) in train_loader:
    #     print(img)
    #     print(label)
