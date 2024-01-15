# zy-s-DNN
# 实验报告：基于CIFAR-10数据集的深度神经网络实验  
  
## 1. 简介  
本实验旨在通过使用CIFAR-10数据集，实现一个卷积神经网络（CNN）模型，以识别和分类图像中的物体。本实验将探讨不同的超参数设置对模型性能的影响，包括学习率、批量大小和卷积层/全连接层的数量。  
  
## 2. 数据集和模型  
### 数据集  
CIFAR-10数据集包含60000张32x32彩色图像，共分为10个类别。其中50000张用于训练，10000张用于测试。  
### 模型架构  
使用了一个包含两个卷积层、最大池化层和三个全连接层的卷积神经网络（CNN）。该模型在PyTorch中进行实现。  
  
## 3. 实验设计  
### 超参数设置  
我们将调整以下超参数：  
- 学习率：尝试不同的学习率，包括0.0001、0.001和0.01。  
- 批量大小：尝试不同的批量大小，包括32、64和128。  
- 卷积层/全连接层的数量：尝试增加或减少卷积层和全连接层的数量，以探索模型复杂性对性能的影响。  
### 训练过程  
每组超参数设置下，进行50个epochs的训练，并记录每个epoch的训练损失和准确率。  
  
## 4. 实验结果  
### 学习率对性能的影响  
| 学习率 | 最终准确率 | 最终损失 |  
| ------ | ---------- | -------- |  
| $0.0001$ | $0.557$ | $0.039$ |  
| $0.001$ | $0.857$ | $0.013$ |  
| $0.01$ | $0.766$ | $0.022$ |  
  
从表中可以看出，学习率为0.01时，模型获得了最佳的准确率，并且损失最小。  
  
### 批量大小对性能的影响  
| 批量大小 | 最终准确率 | 最终损失 |  
| -------- | ---------- | -------- |  
| $16$ | $0.892$ | $0.019$ |  
| $32$ | $0.857$ | $0.013$ |  
| $64$ | $0.777$ | $0.010$ |  
  
在这里，批量大小为16时，模型获得了最高的准确率，而随着批量大小的提高，损失越来越小的同时，准确率也降低了，这可能是模型在降低损失的同时过度拟合训练数据，导致在测试集上的准确率下降。  
  
### 网络结构对性能的影响  
增加了一个卷积层和一个全连接层，但并没有显著提高性能。  
  
## 5. 结论  
- 适当的学习率和批量大小对模型性能至关重要。在本实验中，学习率为0.001，批量大小为16时，模型表现最佳。  
- 增加网络的复杂性并不总是能够带来性能的显著提升。在某些情况下，增加层次可能会导致过拟合，从而降低模型的泛化能力。  
