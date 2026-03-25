import os
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
import warnings
from typing import Dict,Optional, Any

warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = str(42)

# ================= 自适应阈值优化器 =================
class AdaptiveThresholdOptimizer:
    """为每个类别找到最优阈值"""

    def __init__(self, strategy: Optional[Dict[str, Any]] = None):
        self.strategy = strategy or {}                      #储阈值优化策略（如用F1找最佳阈值），如果没传就用空字典
        self.thresholds: Dict[int, float] = {}              #存储每个类别的最佳分类阈值，键是类别ID，值是阈值（如 {0: 0.5, 1: 0.3}）
        self.class_stats: Dict[int, Dict[str, float]] = {}  #存储每个类别的统计信息（如F1、精确率、召回率等），键是类别ID，值是统计指标字典

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray,
            class_names: Dict[int, str]) -> Dict[int, float]:
        """基于验证集找到最优阈值"""
        print("\n🔍 计算自适应阈值...")

        #遍历每个攻击类型，为每个类别单独计算最佳阈值
        for class_id, class_name in class_names.items():
            #将当前类别设为正类（1），其他类别设为负类（0），转为二分类问题
            y_binary = (y_true == class_id).astype(int)
            #取出当前类别的预测概率（模型认为它是该类别的概率）
            y_score = y_prob[:, class_id]

            #获取优化策略（默认用F1分数）
            strategy = self.strategy.get(class_name,
                                         self.strategy.get('default', {'target': 'f1'}))

            #计算精确率-召回率曲线：返回不同阈值下的精确率、召回率和阈值列表
            precision, recall, thresholds = precision_recall_curve(y_binary, y_score)

            #计算每个阈值下的 F1 分数（F1 = 2 * P * R / (P + R)）
            #thresholds 比 precision 少一个，所以用 precision[:-1]
            f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)

            #如果没有阈值（数据有问题），设置默认阈值为0.5
            if len(thresholds) == 0:
                self.thresholds[class_id] = 0.5
                continue

            #找到使 F1 分数最大的索引位置
            best_idx = np.argmax(f1_scores) if len(f1_scores) > 0 else 0
            #取出对应的最佳阈值，如果索引超出范围则用0.5
            self.thresholds[class_id] = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

            #打印该类的最佳阈值
            print(f"   {class_name}: threshold={self.thresholds[class_id]:.4f}")

        #返回所有类别的最佳阈值字典
        return self.thresholds


    def predict(self, probabilities: torch.Tensor) -> torch.Tensor:
        """应用自适应阈值进行预测"""
        #如果没有保存任何阈值，就用默认方法（取概率最大的类别）
        if not self.thresholds:
            return probabilities.argmax(dim=1)

        #获取批次大小和类别数量
        batch_size, num_classes = probabilities.shape
        #初始化预测结果张量，全0
        predictions = torch.zeros(batch_size, dtype=torch.long)

        #将阈值字典转换为张量，方便计算
        #没有指定阈值的类别默认用0.5
        threshold_tensor = torch.tensor(
            [self.thresholds.get(i, 0.5) for i in range(num_classes)],
            device=probabilities.device
        )

        #逐条判断每条边的预测结果
        for i in range(batch_size):
            #当前样本的概率分布
            probs = probabilities[i]

            #找出哪些类别的概率超过了它们的阈值
            mask = probs >= threshold_tensor

            #如果有超过阈值的类别
            if mask.any():
                #复制一份概率
                valid_probs = probs.clone()
                #把没超过阈值的类别的概率设为负无穷（这样它们永远不会被选中）
                valid_probs[~mask] = -float('inf')
                #选概率最大的那个类别作为预测结果
                predictions[i] = valid_probs.argmax()
            else:
                #如果所有概率都没超过阈值，就退回到默认方法（取最大概率）
                predictions[i] = probs.argmax()

        return predictions