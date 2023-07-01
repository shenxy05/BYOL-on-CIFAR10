import numpy as np
import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str, help="Path to pre-trained model (e.g. model-10.pt)")
    
    args = parser.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    test_dataset_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    #准备测试数据集
    test_data = torchvision.datasets.CIFAR10(root='./datasets',train=False,transform=test_dataset_transform
                                         ,download=True)
    test_dataloader = DataLoader(test_data,batch_size=128)
    test_data_size = len(test_data)
    model = torchvision.models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10) # 修改resnet中全连接层的输出类别数
    model.conv1 = nn.Conv2d(3, 64, 5, stride=1, padding=2, bias=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device).cpu().state_dict())
    model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.cuda()
    
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