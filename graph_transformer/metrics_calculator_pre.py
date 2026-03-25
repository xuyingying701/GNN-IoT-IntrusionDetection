import os
import numpy as np
from sklearn.metrics import ( f1_score, accuracy_score,roc_auc_score,average_precision_score)
from sklearn.metrics import precision_score, recall_score
import warnings
from typing import Dict,Any

warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = str(42)

# ================= 评估指标计算器 =================
class MetricsCalculator:
    """全面的评估指标计算"""

    @staticmethod
    def calculate_all(y_true: np.ndarray, y_pred: np.ndarray,
                      y_prob: np.ndarray, class_names: Dict[int, str]) -> Dict[str, Any]:
        """计算所有指标"""
        #字典存储所有计算结果
        metrics = {'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),      #宏平均F1：每个类别F1的算术平均，适合不平衡数据集
                   'micro_f1': f1_score(y_true, y_pred, average='micro', zero_division=0),      #微平均F1：按样本总数计算，适合平衡数据集
                   'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),#加权平均F1：按每个类别的样本数加权，反映整体表现
                   'accuracy': accuracy_score(y_true, y_pred), 'per_class': {}}                 #准确率：正确预测的样本数 / 总样本数

        # ================= 每类指标 =================
        # 获取所有出现过的类别（真实标签和预测标签的并集）
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))

        for label in unique_labels:  # 遍历每个类别
            #获取该类别的名称（如果class_names中没有，则用默认名称）
            class_name = class_names.get(label, f'Class_{label}')
            #计算精确率：预测为该类且正确的样本数 / 预测为该类的样本数
            precision = precision_score(y_true, y_pred, labels=[label], average=None, zero_division=0)[0]
            #计算召回率：预测为该类且正确的样本数 / 实际为该类的样本数
            recall = recall_score(y_true, y_pred, labels=[label], average=None, zero_division=0)[0]
            #计算F1分数：精确率和召回率的调和平均数
            f1 = f1_score(y_true, y_pred, labels=[label], average=None, zero_division=0)[0]

            # 存储该类别的指标
            metrics['per_class'][class_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }

        # ================= AUC-ROC 和 AUC-PR =================
        try:
            # 导入标签二值化工具（将类别标签转换为one-hot编码）
            from sklearn.preprocessing import label_binarize
            # 将真实标签转换为二值化矩阵 [n_samples, n_classes]
            y_true_bin = label_binarize(y_true, classes=unique_labels)

            metrics['auc_roc'] = {}  # 存储每个类别的ROC-AUC
            metrics['auc_pr'] = {}  # 存储每个类别的PR-AUC
            auc_roc_list = []  # 存储所有类别的ROC-AUC，用于计算平均值
            auc_pr_list = []  # 存储所有类别的PR-AUC，用于计算平均值

            for i, label in enumerate(unique_labels):  # 遍历每个类别
                class_name = class_names.get(label, f'Class_{label}')  # 获取类别名称

                # 检查该类别的样本是否同时有正例和负例（至少2个类别）
                if len(np.unique(y_true_bin[:, i])) > 1:
                    # 计算ROC-AUC：模型区分正负类的能力
                    auc_roc = roc_auc_score(y_true_bin[:, i], y_prob[:, label])
                    metrics['auc_roc'][class_name] = float(auc_roc)
                    auc_roc_list.append(auc_roc)

                # 计算PR-AUC：精确率-召回率曲线下的面积，适合不平衡数据
                auc_pr = average_precision_score(y_true_bin[:, i], y_prob[:, label])
                metrics['auc_pr'][class_name] = float(auc_pr)
                auc_pr_list.append(auc_pr)

            # 计算所有类别的平均ROC-AUC
            metrics['mean_auc_roc'] = float(np.mean(auc_roc_list)) if auc_roc_list else 0
            # 计算所有类别的平均PR-AUC
            metrics['mean_auc_pr'] = float(np.mean(auc_pr_list)) if auc_pr_list else 0

        except Exception as e:              #如果计算AUC时出错（如类别不足）
            print(f"   AUC计算跳过: {e}")

        return metrics                      #返回包含所有指标的字典