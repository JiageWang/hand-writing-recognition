import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pickle
import numpy as np
from hwdb import HWDB
from convnet import ConvNet


def train(net,
          criterion,
          optimizer,
          train_loader,
          test_loarder,
          epoch=10,
          save_path='./pretrained_models/'):
    def adjust_learning_rate(optimizer, decay_rate=.9):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
    print("开始训练...")
    net.train()
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    for epoch in range(epoch):
        sum_loss = 0.0
        total = 0
        correct = 0
        if epoch/3 == 1:
            adjust_learning_rate(optimizer, 0.5)
        # 数据读取
        for i, (inputs, labels) in enumerate(train_loader):
            # 梯度清零
            optimizer.zero_grad()

            # forward + backward
            if torch.cuda.is_available():
                # inputs, labels = Variable(inputs.cuda(0)), Variable(labels.cuda(0))
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
                #print(inputs.device)
            else:
                print('cuda not available')
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

            loss.backward()
            optimizer.step()

            #print(loss.item())
            # 每训练100个batch打印一次平均loss与acc
            sum_loss += loss.item()
            # if i % 100 == 99:
            if i % 100 == 99:
                loss = sum_loss/100
                print('epoch: %d, batch: %d loss: %.03f'
                      % (epoch + 11, i + 1, loss), end=',')
                # 每跑完一次epoch测试一下准确率
                acc = 100 * correct / total
                print('acc：%d%%' % (acc))
                total = 0
                correct = 0
                sum_loss = 0.0

        print("epoch%d 训练结束, 正在保存模型..."%(epoch+11))
        torch.save(net.state_dict(), save_path+'handwriting_iter_%03d.pth' % (epoch + 11))
        if epoch%3 == 0:
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:
                    images, labels = images.to('cuda'), labels.to('cuda')
                    outputs = net(images)
                    # 取得分最高的那个类
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                print('correct number: ',correct)
                print('totol number:', total)
                acc = 100 * correct / total
                print('第%d个epoch的识别准确率为：%d%%' % (epoch+11, acc))



if __name__ == "__main__":
    # 超参数
    batch_size = 100

    # 读取分类类别
    f = open('char_dict', 'rb')
    class_dict = pickle.load(f)
    num_classes = len(class_dict)

    # 读取数据
    dataset = HWDB()
    print("训练集数据:", dataset.train_size)
    print("测试集数据:", dataset.test_size)
    train_loader, test_loader = dataset.get_loader(batch_size)


    net = ConvNet(num_classes)
    print('网络结构：\n', net)
    if torch.cuda.is_available():
        net = net.cuda(0)
    else:
        print('cuda not available')
    net.load_state_dict(torch.load('./pretrained_models/handwriting_iter_010.pth'))
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    optimizer = optim.RMSprop(net.parameters(), lr=0.000005, momentum=0.9, weight_decay=0.0005)
    train(net, criterion, optimizer, train_loader, test_loader)


