import os
import random
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt


class HWDB(object):
    def __init__(self,path, transform):
        # 预处理过程

        traindir = os.path.join(path, 'train')
        testdir = os.path.join(path, 'test')

        self.trainset = datasets.ImageFolder(traindir, transform)
        self.testset = datasets.ImageFolder(testdir, transform)
        self.train_size = len(self.trainset)
        self.test_size = len(self.testset)
        self.num_classes = len(self.trainset.classes)
        self.class_to_idx = self.trainset.class_to_idx

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
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    dataset = HWDB(path=r'data', transform=transform)
    print("训练集数量：", dataset.train_size)
    print("测试集数量：", dataset.test_size)
    print("类别数量：", dataset.num_classes)
    index = random.randint(0, dataset.train_size)
    img = dataset.get_sample(index)[0][0]
    plt.imshow(img, cmap='gray')
    plt.show()
