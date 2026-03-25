import os
import numpy as np
from sklearn.metrics import (f1_score, accuracy_score, roc_auc_score,
                             average_precision_score, confusion_matrix,
                             precision_score, recall_score)
import warnings
from typing import Dict, Any

warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = str(42)


class MetricsCalculator:
    """全面的评估指标计算器 - 修复多分类FPR/FNR"""

    @staticmethod
    def calculate_all(y_true: np.ndarray, y_pred: np.ndarray,
                      y_prob: np.ndarray, class_names: Dict[int, str]) -> Dict[str, Any]:
        """计算所有指标 - 包括正确的多分类FPR/FNR"""

        # 基础指标
        metrics = {
            'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'micro_f1': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
            'per_class': {}
        }

        # 获取所有类别
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))

        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

        # ===== 修复：计算多分类宏平均FPR/FNR =====
        fpr_list = []
        fnr_list = []
        per_class_metrics = {}

        for i, label in enumerate(unique_labels):
            class_name = class_names.get(label, f'Class_{label}')

            # 基础指标
            precision = precision_score(y_true, y_pred, labels=[label],
                                        average=None, zero_division=0)[0]
            recall = recall_score(y_true, y_pred, labels=[label],
                                  average=None, zero_division=0)[0]
            f1 = f1_score(y_true, y_pred, labels=[label],
                          average=None, zero_division=0)[0]

            # 混淆矩阵元素
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp

            # FPR/FNR
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            fpr_list.append(fpr)
            fnr_list.append(fnr)

            per_class_metrics[class_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'fpr': float(fpr),
                'fnr': float(fnr),
                'tp': int(tp), 'fn': int(fn),
                'fp': int(fp), 'tn': int(tn),
                'support': int(tp + fn)
            }

            metrics['per_class'][class_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }

        # 宏平均FPR/FNR
        metrics['macro_fpr'] = float(np.mean(fpr_list))
        metrics['macro_fnr'] = float(np.mean(fnr_list))
        metrics['per_class_fpr_fnr'] = per_class_metrics

        # 加权平均（按样本数加权）
        supports = [per_class_metrics[class_names.get(i, f'Class_{i}')]['support']
                    for i in unique_labels]
        metrics['weighted_fpr'] = float(np.average(fpr_list, weights=supports))
        metrics['weighted_fnr'] = float(np.average(fnr_list, weights=supports))

        # ===== 保留：二分类视角（Normal vs Attack）=====
        normal_class_id = None
        for idx, name in class_names.items():
            if name.lower() == 'normal':
                normal_class_id = idx
                break

        if normal_class_id is not None and normal_class_id in unique_labels:
            normal_idx = list(unique_labels).index(normal_class_id)

            # 二分类视角：Attack=正类, Normal=负类
            tp_bin = sum(cm[i, i] for i in range(len(unique_labels)) if i != normal_idx)
            fn_bin = cm[normal_idx, :].sum() - cm[normal_idx, normal_idx]  # 攻击被预测为Normal
            fp_bin = cm[:, normal_idx].sum() - cm[normal_idx, normal_idx]  # Normal被预测为攻击
            tn_bin = cm[normal_idx, normal_idx]

            fpr_bin = fp_bin / (fp_bin + tn_bin) if (fp_bin + tn_bin) > 0 else 0.0
            fnr_bin = fn_bin / (fn_bin + tp_bin) if (fn_bin + tp_bin) > 0 else 0.0

            metrics['binary_fpr'] = float(fpr_bin)
            metrics['binary_fnr'] = float(fnr_bin)

            print(f"\n[二分类视角 - Normal vs Attack]")
            print(f"   误报率 (FPR): {fpr_bin:.4f} ({fp_bin}/{fp_bin + tn_bin})")
            print(f"   漏报率 (FNR): {fnr_bin:.4f} ({fn_bin}/{fn_bin + tp_bin})")

        # 打印多分类指标
        print(f"\n[多分类视角 - 宏平均]")
        print(f"   宏平均误报率 (Macro-FPR): {metrics['macro_fpr']:.4f} ({metrics['macro_fpr']:.2%})")
        print(f"   宏平均漏报率 (Macro-FNR): {metrics['macro_fnr']:.4f} ({metrics['macro_fnr']:.2%})")
        print(f"   加权误报率 (Weighted-FPR): {metrics['weighted_fpr']:.4f}")
        print(f"   加权漏报率 (Weighted-FNR): {metrics['weighted_fnr']:.4f}")

        print(f"\n[各类别FPR/FNR详情]")
        for class_name, values in per_class_metrics.items():
            if values['fnr'] > 0.05 or values['fpr'] > 0.01:
                print(f"   {class_name:12s}: FNR={values['fnr']:>6.2%}, FPR={values['fpr']:>6.2%}, "
                      f"Recall={values['recall']:>6.2%}")

        # AUC计算（保持原有）
        try:
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=unique_labels)

            metrics['auc_roc'] = {}
            metrics['auc_pr'] = {}
            auc_roc_list, auc_pr_list = [], []

            for i, label in enumerate(unique_labels):
                class_name = class_names.get(label, f'Class_{label}')

                if len(np.unique(y_true_bin[:, i])) > 1:
                    auc_roc = roc_auc_score(y_true_bin[:, i], y_prob[:, label])
                    metrics['auc_roc'][class_name] = float(auc_roc)
                    auc_roc_list.append(auc_roc)

                auc_pr = average_precision_score(y_true_bin[:, i], y_prob[:, label])
                metrics['auc_pr'][class_name] = float(auc_pr)
                auc_pr_list.append(auc_pr)

            metrics['mean_auc_roc'] = float(np.mean(auc_roc_list)) if auc_roc_list else 0
            metrics['mean_auc_pr'] = float(np.mean(auc_pr_list)) if auc_pr_list else 0

        except Exception as e:
            print(f"   AUC计算跳过: {e}")

        return metrics