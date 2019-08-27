import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
from torchvision import transforms
from torchsummary import summary

from hwdb import HWDB
from model import ConvNet


def valid(epoch, net, test_loarder, writer):
    print("epoch %d 开始验证..." % epoch)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loarder:
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            # 取得分最高的那个类
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('correct number: ', correct)
        print('totol number:', total)
        acc = 100 * correct / total
        print('第%d个epoch的识别准确率为：%d%%' % (epoch, acc))
        writer.add_scalar('valid_acc', acc, global_step=epoch)


def train(epoch, net, criterion, optimizer, train_loader, writer, save_iter=100):
    print("epoch %d 开始训练..." % epoch)
    net.train()
    sum_loss = 0.0
    total = 0
    correct = 0
    # 数据读取
    for i, (inputs, labels) in enumerate(train_loader):
        # 梯度清零
        optimizer.zero_grad()
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # 取得分最高的那个类
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss.backward()
        optimizer.step()

        # 每训练100个batch打印一次平均loss与acc
        sum_loss += loss.item()
        if (i + 1) % save_iter == 0:
            batch_loss = sum_loss / save_iter
            # 每跑完一次epoch测试一下准确率
            acc = 100 * correct / total
            print('epoch: %d, batch: %d loss: %.03f, acc: %.04f'
                  % (epoch, i + 1, batch_loss, acc))
            writer.add_scalar('train_loss', batch_loss, global_step=i + len(train_loader) * epoch)
            writer.add_scalar('train_acc', acc, global_step=i + len(train_loader) * epoch)
            for name, layer in net.named_parameters():
                writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(),
                                     global_step=i + len(train_loader) * epoch)
                writer.add_histogram(name + '_data', layer.cpu().data.numpy(),
                                     global_step=i + len(train_loader) * epoch)
            total = 0
            correct = 0
            sum_loss = 0.0


if __name__ == "__main__":
    # 超参数
    epochs = 20
    batch_size = 100
    lr = 0.01

    data_path = r'data'
    log_path = r'logs/batch_{}_lr_{}'.format(batch_size, lr)
    save_path = r'checkpoints/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 读取分类类别
    with open('char_dict', 'rb') as f:
        class_dict = pickle.load(f)
    num_classes = len(class_dict)

    # 读取数据
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    dataset = HWDB(path=data_path, transform=transform)
    print("训练集数据:", dataset.train_size)
    print("测试集数据:", dataset.test_size)
    trainloader, testloader = dataset.get_loader(batch_size)

    net = ConvNet(num_classes)
    if torch.cuda.is_available():
        net = net.cuda()
    # net.load_state_dict(torch.load('checkpoints/handwriting_iter_004.pth'))

    print('网络结构：\n')
    summary(net, input_size=(3, 64, 64), device='cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)
    writer = SummaryWriter(log_path)
    for epoch in range(epochs):
        train(epoch, net, criterion, optimizer, trainloader, writer=writer)
        valid(epoch, net, testloader, writer=writer)
        print("epoch%d 结束, 正在保存模型..." % epoch)
        torch.save(net.state_dict(), save_path + 'handwriting_iter_%03d.pth' % epoch)
