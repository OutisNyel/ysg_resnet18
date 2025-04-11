import torch.nn as nn
import torch
from torchvision import models

# 模型定义
class DualResNet(nn.Module):
    def __init__(self, label_num=7):
        super(DualResNet, self).__init__()
        from torchvision.models import ResNet18_Weights
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 使用weights参数加载预训练权重
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 512)  # 修改ResNet18的全连接层
        self.classifier = nn.Linear(1024, label_num)

    def forward(self, x):
        # 前向传播函数
        (left_input, right_input) = x
        left_features = self.resnet(left_input)  # 提取左眼图像的特征
        right_features = self.resnet(right_input)  # 提取右眼图像的特征
        combined_features = torch.cat((left_features, right_features), dim=1)  # 拼接左右眼的特征
        output = self.classifier(combined_features)  # 使用全连接层输出最终的预测
        return output

