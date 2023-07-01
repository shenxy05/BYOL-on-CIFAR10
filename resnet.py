import numpy as np
import time
import torch.optim
import torchvision
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torch.utils import tensorboard
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("./runs/resnet18")

 
# 增强数据集transforms
train_dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32,padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
test_dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
 
#准备训练数据集
train_data = torchvision.datasets.CIFAR10(root='./datasets',train=True,transform=train_dataset_transform
                                          ,download=True)
 
#准备测试数据集
test_data = torchvision.datasets.CIFAR10(root='./datasets',train=False,transform=test_dataset_transform
                                         ,download=True)



train_data_size = len(train_data)
test_data_size = len(test_data)
print('训练集的大小为{} \n测试集的大小为{}'.format(train_data_size,test_data_size))
 
#利用Dataloader来加载数据集
train_dataloader = DataLoader(train_data,batch_size=128)
test_dataloader = DataLoader(test_data,batch_size=128)
 

model = torchvision.models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10) # 修改resnet中全连接层的输出类别数
model.conv1 = nn.Conv2d(3, 64, 5, stride=1, padding=2, bias=False)
model = model.cuda()

#损失函数

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
 
#优化器
learning_rate = 0.01
optimizer = torch.optim.Adam(params=model.parameters(),lr=learning_rate,betas=(0.9,0.999),eps=1e-08,weight_decay=0)
#设置学习率衰减
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch:1/(epoch+1))
 
#设置训练网络的一些参数
#记录训练的次数
total_train_step = 0
#记录测试的次数
total_test_step = 0
#训练的轮数
epoch = 50
since = time.time()
for i in range(epoch):
    start = time.time()
    print('-----------第{}轮训练开始-----------'.format(i+1))     #因为batch_size大小为64,50000/64=781.25,故每训练781次就会经过一轮epoch
    print(f"learning rate: {optimizer.param_groups[0]['lr']}")
    #训练步骤开始
    model.train()   #设置模型进入训练状态，仅对dropout,batchnorm...等有作用，如果有就要调用这里模型暂时没有可不调用
    for data in train_dataloader:           #train_dataloader的batch_size为64,从训练的train_dataloader中取数据
        imgs , targets = data            #
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = model(imgs)                  #将img放入神经网络中进行训练
        loss = loss_fn(outputs,targets)     #计算预测值与真实值之间的损失
 
        #优化器优化模型
        optimizer.zero_grad()    #运行前梯度清零
        loss.backward()          #反向传播
        optimizer.step()         #随机梯度下降更新参数
        total_train_step = total_train_step + 1   #训练次数加一
        if total_train_step % 100 == 0:
            print('训练次数:{}，Loss:{}'.format(total_train_step,loss.item()))    #.item()的作用是输出数字，与训练次数格式相同
            writer.add_scalar('train_loss',loss.item(),total_train_step)
 
 
 
    #测试步骤开始
    model.eval()  #设置模型进入验证状态，仅对dropout,batchnorm...等有作用，如果有就要调用这里模型暂时没有可不调用
    total_test_loss = 0
    total_test_accuracy = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = model(imgs)
            loss = loss_fn(outputs,targets)
 
            total_test_loss = total_test_loss + loss.item()   #所有loss的加和，由于total_test_loss是数字，而loss是Tensor数据类型，故加.item()
            accuracy = (outputs.argmax(dim=1) == targets).sum()  #输出每次预测正确的个数
            total_accuracy = total_accuracy + accuracy    #测试集上10000个数据的正确个数总和
 
        print('整体测试集上的loss:{}'.format(total_test_loss))
        print('整体测试集上的正确率：{}'.format(total_accuracy / test_data_size))
        writer.add_scalar('test_loss',total_test_loss,total_test_step)
        writer.add_scalar('test_accuracy',total_accuracy / test_data_size,total_test_step)
        total_test_step = total_test_step + 1
 
        torch.save(model,'./checkpoint/model_{}.pth'.format(i))
        print('模型已保存')
        spent = time.time() - start
        total_spent = time.time() - since
        print('epoch spent: {:.0f}m {:.0f}s || total spent: {:.0f}m {:.0f}s '.format(spent // 60, spent % 60, total_spent // 60, total_spent % 60))
    scheduler.step()

writer.close()
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 