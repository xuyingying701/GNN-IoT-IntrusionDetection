import os
import numpy as np
from sklearn.metrics import (f1_score, accuracy_score, roc_auc_score,average_precision_score, confusion_matrix,precision_score, recall_score)
import warnings
from typing import Dict, Any

warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = str(42)


class MetricsCalculator:
    """全面的评估指标计算器 - 修复多分类FPR/FNR"""

    @staticmethod
    def calculate_all(y_true: np.ndarray, y_pred: np.ndarray,
                      y_prob: np.ndarray, class_names: Dict[int, str]) -> Dict[str, Any]:   #真实标签，预测标签，预测概率，攻击类型ID到名称的映射字典
        """计算所有指标 - 包括正确的多分类FPR/FNR"""

        #基础指标
        metrics = {
            'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),         #宏平均F1：每个类别F1的算术平均，适合不平衡数据集
            'micro_f1': f1_score(y_true, y_pred, average='micro', zero_division=0),         #微平均F1：按样本总数计算，适合平衡数据集
            'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),   #加权平均F1：按每个类别的样本数加权
            'accuracy': accuracy_score(y_true, y_pred),                                     #准确率：正确预测的样本数 / 总样本数
            'per_class': {}                                                                 #每类详细指标，后续循环填充
        }

        #获取唯一的类别列表
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))

        #计算混淆矩阵，labels指定所有类别的顺序，确保矩阵的维度与类别数一致
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

        # ===== 修复：计算多分类宏平均FPR/FNR =====
        fpr_list = []               #每个类别的误报率（False Positive Rate）列表
        fnr_list = []               #每个类别的漏报率（False Negative Rate）列表
        per_class_metrics = {}      #每个类别的详细性能指标的字典

        for i, label in enumerate(unique_labels):                   #遍历所有类别，为每个类别计算性能指标
            class_name = class_names.get(label, f'Class_{label}')   #字典中查找到的类别名称或者默认生成的 Class_数字

            #基础指标
            precision = precision_score(y_true, y_pred, labels=[label],
                                        average=None, zero_division=0)[0]   #精确率（预测为正的样本中，有多少是真的正）
            recall = recall_score(y_true, y_pred, labels=[label],
                                  average=None, zero_division=0)[0]         #召回率（真正的正样本中，有多少被正确预测出来）
            f1 = f1_score(y_true, y_pred, labels=[label],
                          average=None, zero_division=0)[0]                 #F1分数（精确率和召回率的调和平均，综合评估模型性能）

            #混淆矩阵元素
            tp = cm[i, i]                   #真阳性（正确识别的攻击数量）
            fn = cm[i, :].sum() - tp        #假阴性（漏掉的攻击数量）
            fp = cm[:, i].sum() - tp        #假阳性（误报的攻击数量）
            tn = cm.sum() - tp - fn - fp    #真阴性（正确识别的正常行为数量）

            #计算误报率（FPR）和漏报率（FNR）
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            fpr_list.append(fpr)            #把当前类别的误报率加入列表
            fnr_list.append(fnr)            #把当前类别的漏报率加入列表

            per_class_metrics[class_name] = {   #把当前类别的所有性能指标加入字典
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'fpr': float(fpr),
                'fnr': float(fnr),
                'tp': int(tp), 'fn': int(fn),
                'fp': int(fp), 'tn': int(tn),
                'support': int(tp + fn)         #支持数（该类在测试集中的真实样本数）
            }

            metrics['per_class'][class_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)                 #每个类别的精确率、召回率、F1分数存储到 metrics 字典中
            }

        #计算宏平均FPR/FNR
        metrics['macro_fpr'] = float(np.mean(fpr_list))     #所有种类的整体误报率
        metrics['macro_fnr'] = float(np.mean(fnr_list))     #所有种类的整体漏报率
        metrics['per_class_fpr_fnr'] = per_class_metrics    #每个类别的详细指标

        #加权平均（按样本数加权）
        supports = [per_class_metrics[class_names.get(i, f'Class_{i}')]['support']
                    for i in unique_labels]                                         #每个类别的样本数量
        metrics['weighted_fpr'] = float(np.average(fpr_list, weights=supports))     #加权平均误报率
        metrics['weighted_fnr'] = float(np.average(fnr_list, weights=supports))     #加权平均漏报率

        #====== 二分类视角（Normal vs Attack）======
        normal_class_id = None                  #"正常流量"类别对应的ID
        for idx, name in class_names.items():   #找"正常流量"类别对应的ID
            if name.lower() == 'normal':
                normal_class_id = idx
                break

        if normal_class_id is not None and normal_class_id in unique_labels:    #如果找到了正常类别，并且正常类别的ID在当前计算的类别列表中
            normal_idx = list(unique_labels).index(normal_class_id)         #正常类别在 unique_labels 列表中的索引位置

            #二分类视角：Attack=正类, Normal=负类
            tp_bin = sum(cm[i, i] for i in range(len(unique_labels)) if i != normal_idx)#真阳性
            fn_bin = cm[normal_idx, :].sum() - cm[normal_idx, normal_idx]               #假阴性
            fp_bin = cm[:, normal_idx].sum() - cm[normal_idx, normal_idx]               #假阳性
            tn_bin = cm[normal_idx, normal_idx]                                         #真阴性

            fpr_bin = fp_bin / (fp_bin + tn_bin) if (fp_bin + tn_bin) > 0 else 0.0  #二分类视角下的误报率
            fnr_bin = fn_bin / (fn_bin + tp_bin) if (fn_bin + tp_bin) > 0 else 0.0  #二分类视角下的漏报率
            #将二分类视角下的误报率和漏报率存储到结果字典 metrics 中
            metrics['binary_fpr'] = float(fpr_bin)
            metrics['binary_fnr'] = float(fnr_bin)

            print(f"\n[二分类视角 - Normal vs Attack]")
            print(f"   误报率 (FPR): {fpr_bin:.4f} ({fp_bin}/{fp_bin + tn_bin})")
            print(f"   漏报率 (FNR): {fnr_bin:.4f} ({fn_bin}/{fn_bin + tp_bin})")

        #打印多分类指标
        print(f"\n[多分类视角 - 宏平均]")
        print(f"   宏平均误报率 (Macro-FPR): {metrics['macro_fpr']:.4f} ({metrics['macro_fpr']:.2%})")
        print(f"   宏平均漏报率 (Macro-FNR): {metrics['macro_fnr']:.4f} ({metrics['macro_fnr']:.2%})")
        print(f"   加权误报率 (Weighted-FPR): {metrics['weighted_fpr']:.4f}")
        print(f"   加权漏报率 (Weighted-FNR): {metrics['weighted_fnr']:.4f}")

        #打印每个类别的误报率和漏报率详情，特别关注检测效果较差的类别（漏报率>5%或误报率>1%）
        print(f"\n[各类别FPR/FNR详情]")
        for class_name, values in per_class_metrics.items():
            if values['fnr'] > 0.05 or values['fpr'] > 0.01:
                print(f"   {class_name:12s}: FNR={values['fnr']:>6.2%}, FPR={values['fpr']:>6.2%}, "
                      f"Recall={values['recall']:>6.2%}")

        #AUC计算（保持原有）  AUC = Area Under the Curve（曲线下面积
        try:
            from sklearn.preprocessing import label_binarize        #尝试导入label_binarize函数，用于将多分类标签转换为二分类格式

            #将真实标签转换为二值化格式，每列对应一个类别，是该类为1，不是为0
            y_true_bin = label_binarize(y_true, classes=unique_labels)  #将多分类真实标签转换成的二值化矩阵，例如：y_true=[0,1,2] → [[1,0,0],[0,1,0],[0,0,1]]

            metrics['auc_roc'] = {}             #初始化存储每个类别AUC-ROC的字典
            metrics['auc_pr'] = {}              #初始化存储每个类别AUC-PR的字典
            auc_roc_list, auc_pr_list = [], []  #初始化列表，用于收集所有类别的AUC值，后续计算平均值

            for i, label in enumerate(unique_labels):   #遍历每个类别，i是索引，label是类别ID
                class_name = class_names.get(label, f'Class_{label}')   #根据类别ID获取类别名称，如果不存在则使用默认名称Class_{label}

                if len(np.unique(y_true_bin[:, i])) > 1:            #检查该类别在测试集中是否既有正例又有负例，如果只有一种样本（全是攻击或全是正常），则无法计算AUC
                    auc_roc = roc_auc_score(y_true_bin[:, i], y_prob[:, label])     #计算ROC曲线下面积（AUC-ROC），评估模型区分该类与其他类的能力
                    metrics['auc_roc'][class_name] = float(auc_roc)                 #存储该类的AUC-ROC，转换为float确保JSON序列化兼容
                    auc_roc_list.append(auc_roc)                                    #将该类AUC加入列表，用于后续计算平均值

                auc_pr = average_precision_score(y_true_bin[:, i], y_prob[:, label])#计算PR曲线下面积（AUC-PR），在类别不平衡时比AUC-ROC更可靠
                metrics['auc_pr'][class_name] = float(auc_pr)       #存储该类的AUC-PR
                auc_pr_list.append(auc_pr)                          #将该类AUC-PR加入列表

            metrics['mean_auc_roc'] = float(np.mean(auc_roc_list)) if auc_roc_list else 0   #计算所有类别的平均AUC-ROC，如果没有有效类别则设为0
            metrics['mean_auc_pr'] = float(np.mean(auc_pr_list)) if auc_pr_list else 0      #计算所有类别的平均AUC-PR

        except Exception as e:      #如果计算过程中出现异常（如缺少依赖库），打印提示并跳过AUC计算
            print(f"   AUC计算跳过: {e}")

        return metrics