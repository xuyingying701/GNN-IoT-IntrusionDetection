import os
import torch
import torch.nn.functional as F
import warnings
from typing import Dict,Optional

warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = str(42)

# ================= 优化版Focal Loss（带标签平滑） =================
class FocalLoss(torch.nn.Module):
    """优化版Focal Loss - 支持标签平滑"""

    def __init__(self, weight: Optional[torch.Tensor] = None,
                 class_gamma: Optional[Dict[int, float]] = None,
                 default_gamma: float = 2.0,        #如果调用时没传这个参数，就用 2.0
                 label_smoothing: float = 0.1):
        super().__init__()                      #调用父类初始化
        self.weight = weight                    #类别权重，处理样本不平衡（少数类权重更高）
        self.class_gamma = class_gamma or {}    #各类别的gamma值，处理不同攻击的检测难度（难检测的攻击gamma更大）
        self.default_gamma = default_gamma      #默认gamma值，用于没有单独设置gamma的类别
        self.label_smoothing = label_smoothing  #标签平滑系数，防止模型过自信，提高泛化能力

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        #标签平滑
        n_classes = input.size(1)                   #获取类别数量（攻击类型数，比如10）  [10000,10]
        log_probs = F.log_softmax(input, dim=1)     #将原始分数转换为对数概率（log概率）  [10000,10]

        #创建平滑标签 [10000,10]
        with torch.no_grad():                       #在这个代码块内，不计算梯度（这些操作不需要反向传播）
            smooth_targets = torch.full_like(log_probs, self.label_smoothing / (n_classes - 1))    #创建和log_probs形状相同、所有值都是self.label_smoothing / (n_classes - 1)的张量
            smooth_targets.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)     #在正确类别的列位置，填入 1.0 - label_smoothing

        #计算带标签平滑的交叉熵损失
        ce = -(smooth_targets * log_probs).sum(dim=1) #CE = -Σ(smooth_targets * log(p))

        #应用Focal权重
        gamma_per_sample = torch.full_like(target, self.default_gamma, dtype=torch.float)#每个样本的 gamma 值，创建和 target 形状相同的张量，全部填充为默认 gamma 值（如 2.0）
        for class_id, gamma in self.class_gamma.items():
            mask = (target == class_id)     #找出哪些样本属于当前攻击类型
            gamma_per_sample[mask] = gamma  #把这些样本的 gamma 值替换为该攻击类型的特定 gamma

        pt = torch.exp(-ce)                             #计算模型对正确类别的预测概率
        focal_weight = (1 - pt) ** gamma_per_sample     #计算每个样本的Focal权重

        #应用类别权重
        if self.weight is not None:             #如果设置了类别权重
            sample_weight = self.weight[target] #根据每个样本的真实标签，取出对应的类别权重
            loss = (focal_weight * ce * sample_weight).mean()   #三个因子相乘，再取平均
        else:                                   #如果没有设置类别权重
            loss = (focal_weight * ce).mean()   #只乘Focal权重，再取平均

        return loss             #返回最终损失值（一个数字）