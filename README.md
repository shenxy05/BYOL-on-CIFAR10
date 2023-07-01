## 监督学习与自监督学习（BYOL）在CIFAR-10上的性能对比

- 自监督预训练：

  ```
  python main.py
  ```

- 使用自监督学习到的特征进行线性分类：

  ```
  python linear_evaluation.py --model_path ./checkpoint/BYOL/model-final.pt
  ```

- 直接用ResNet18进行监督学习：

  ```
  python resnet.py
  ```

- 使用训练好的ResNet18模型进行预测：

  ```
  python test_resnet.py --model_path ./checkpoint/resnet18/model_49.pth
  ```



#### 模型参数下载

下载报告中的模型之后，将checkpoint和runs文件夹放在主目录中
