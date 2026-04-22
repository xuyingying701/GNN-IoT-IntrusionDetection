"""
================================================================================
FairGraphSAGE Baseline - 直接复用主模型模块的精简版
================================================================================
唯一差异：用 SAGEConv 替代 MAF 模块
其他所有组件直接从主模型导入，确保100%一致
================================================================================
"""

import os
import sys
import gc
import warnings
import time
import copy
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

# ==================== 添加主模型路径 ====================
# 假设主模型代码在当前目录或上级目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ==================== 环境配置 ====================
if sys.platform == 'win32':
    try:
        import torch
        if not torch.cuda.is_available():
            os.environ["OMP_NUM_THREADS"] = "4"
            os.environ["MKL_NUM_THREADS"] = "4"
    except:
        pass

import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')

# 固定随机种子
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# ==================== 直接导入主模型的所有组件 ====================
from config import Config
from data_processor import DataProcessor
from graph_builder import GraphBuilder
from focal_loss import FocalLoss
from trainer import Trainer
from edge_batch_loader import EdgeBatchLoader
from adaptive_threshold_optimizer import AdaptiveThresholdOptimizer
from metrics_calculator import MetricsCalculator

print("=" * 80)
print("🚀 FairGraphSAGE Baseline - 复用主模型组件")
print("=" * 80)


# ==================== FairGraphSAGE 模型（唯一修改的部分）====================
class FairGraphSAGE(nn.Module):
    """
    与 GraphTransformer 严格对齐的 GraphSAGE 基线
    复用主模型的编码器和分类器结构，仅替换卷积层
    """

    def _init__(self, in_ch: int, edge_ch: int, hidden: int, out_ch: int,
                 heads: int = 4, layers: int = 2, dropout: float = 0.3):
        super().__init__()

        # ===== 节点编码器（与 GraphTransformer 完全一致）=====
        self.node_encoder = nn.Sequential(
            nn.Linear(in_ch, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ===== 边编码器（与 GraphTransformer 完全一致）=====
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_ch, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ===== 图卷积层（唯一差异：SAGEConv 替代 MAF）=====
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(SAGEConv(hidden, hidden))
            self.norms.append(nn.LayerNorm(hidden))

        self.dropout = nn.Dropout(dropout)

        # ===== 分类器（与 GraphTransformer 完全一致）=====
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden, out_ch)
        )

        self._init_weights()

    def _init_weights(self):
        """与主模型一致的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        # 编码
        x = self.node_encoder(x)
        edge_feat = self.edge_encoder(edge_attr)

        # 图卷积（SAGEConv 替代 MAF）
        for i, conv in enumerate(self.convs):
            x_res = x
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.norms[i](x + x_res)  # 残差连接 + LayerNorm

        # 边分类（与主模型一致）
        src, dst = edge_index
        edge_out = torch.cat([x[src], x[dst], edge_feat], dim=1)
        return self.classifier(edge_out)


# ==================== 主函数 ====================
def main():
    # ===== 使用与主模型完全相同的配置 =====
    config = Config()
    config.__post_init__()

    # 修改输出目录，避免覆盖主模型结果
    config.output_dir = "D:\\01Thesis\\04Git_project\\results"
    config.__post_init__()  #重新创建输出目录

    print("=" * 80)
    print("🚀 FairGraphSAGE - 复用主模型组件，仅替换卷积层")
    print("=" * 80)
    print(f"数据路径: {config.data_path}")
    print(f"输出目录: {config.run_dir}")
    print(f"隐藏层维度: {config.hidden_channels}")
    print(f"学习率: {config.learning_rate}")
    print(f"Dropout: {config.dropout}")
    print("=" * 80)

    try:
        t0 = time.time()

        #===== 1. 数据处理（直接复用）=====
        processor = DataProcessor(config)
        builder = GraphBuilder()

        df = processor.load_data()
        df, train_idx, val_idx, test_idx, attack_names, feature_cols = processor.preprocess(df)
        data = builder.build(df, train_idx, val_idx, test_idx, feature_cols)

        del df
        gc.collect()

        # ===== 2. 类别权重（与主模型完全一致）=====
        y_train = data.y[data.train_mask].numpy()
        present_classes = np.unique(y_train)
        class_weights_dict = compute_class_weight('balanced', classes=present_classes, y=y_train)

        full_class_weights = np.ones(len(attack_names), dtype=np.float32)
        for i, cls in enumerate(present_classes):
            full_class_weights[cls] = class_weights_dict[i] ** 0.7

        attack_name_to_id = {v: k for k, v in attack_names.items()}

        for class_name, boost in config.class_specific_boost.items():
            if class_name in attack_name_to_id:
                class_id = attack_name_to_id[class_name]
                full_class_weights[class_id] *= boost
                print(f"🎯 {class_name} (ID:{class_id}) 权重: {full_class_weights[class_id]:.2f}")

        class_gamma_map = {}
        for class_name, gamma in config.class_gamma.items():
            if class_name in attack_name_to_id:
                class_gamma_map[attack_name_to_id[class_name]] = gamma

        class_weights = torch.tensor(full_class_weights, dtype=torch.float)

        print(f"\n⚖️ 最终类别权重:")
        for i, name in attack_names.items():
            print(f"   {name}: {full_class_weights[i]:.2f}")

        # ===== 3. 设备配置 =====
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n使用设备: {device}")

        data = data.to(device)
        class_weights = class_weights.to(device)

        # ===== 4. 模型初始化（使用 FairGraphSAGE）=====
        model = FairGraphSAGE(
            in_ch=data.x.size(1),
            edge_ch=data.edge_attr.size(1),
            hidden=config.hidden_channels,
            out_ch=len(attack_names),
            heads=config.num_heads,
            layers=config.num_layers,
            dropout=config.dropout
        ).to(device)

        print(f"\n🤖 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

        # ===== 5. 损失函数（直接复用）=====
        criterion = FocalLoss(
            weight=class_weights,
            class_gamma=class_gamma_map,
            default_gamma=config.base_focal_gamma,
            label_smoothing=config.label_smoothing if config.use_label_smoothing else 0.0
        )

        # ===== 6. 优化器和调度器（与主模型一致）=====
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )

        # ===== 7. 训练（直接复用 Trainer）=====
        trainer = Trainer(model, config, attack_names, device)
        model, best_thresholds = trainer.train(data, criterion, optimizer, scheduler)

        # ===== 8. 最终评估（直接复用）=====
        print("\n[4/4] 最终测试集评估...")
        val_loader = EdgeBatchLoader(data, config, shuffle=False)
        test_preds, test_labels, test_probs = trainer.evaluate(val_loader, 'test')

        if test_preds is not None:
            y_true = test_labels.numpy()
            y_pred_base = test_preds.numpy()
            y_prob = test_probs.numpy()

            if config.use_adaptive_threshold and best_thresholds is not None:
                threshold_optimizer = AdaptiveThresholdOptimizer(config.threshold_strategy)
                threshold_optimizer.thresholds = best_thresholds
                y_pred_optimized = threshold_optimizer.predict(test_probs).numpy()
            else:
                y_pred_optimized = y_pred_base

            # 计算指标（直接复用）
            metrics = MetricsCalculator.calculate_all(
                y_true, y_pred_optimized, y_prob, attack_names)

            print("\n" + "=" * 70)
            print(f"🏆 最终 Macro-F1: {metrics['macro_f1']:.4f}")
            print(f"📊 Accuracy: {metrics['accuracy']:.4f}")
            print(f"📈 Mean AUC-ROC: {metrics.get('mean_auc_roc', 0):.4f}")
            print(f"📉 Mean AUC-PR: {metrics.get('mean_auc_pr', 0):.4f}")
            print(f"❌ 宏平均误报率 (Macro-FPR): {metrics.get('macro_fpr', 0):.4f}")
            print(f"⚠️ 宏平均漏报率 (Macro-FNR): {metrics.get('macro_fnr', 0):.4f}")
            print("=" * 70)

            unique_labels = sorted(list(set(y_true) | set(y_pred_optimized)))
            target_names = [attack_names.get(i, f'Class_{i}') for i in unique_labels]
            print("\n分类报告:")
            print(classification_report(y_true, y_pred_optimized, labels=unique_labels,
                                        target_names=target_names, digits=4))

            # ===== 保存结果 =====
            results = {
                'model_type': 'FairGraphSAGE',
                'description': 'Baseline with SAGEConv replacing MAF',
                'config': {k: str(v) if isinstance(v, (type, torch.device)) else v
                          for k, v in config.__dict__.items()},
                'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                           for k, v in metrics.items()},
                'thresholds': {int(k): float(v) for k, v in best_thresholds.items()}
                if best_thresholds else None,
                'training_history': {
                    'loss': trainer.train_losses,
                    'val_f1': trainer.val_f1s,
                    'test_f1': trainer.test_f1s
                }
            }

            results_path = os.path.join(config.run_dir, 'results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

            # 保存模型
            model_path = os.path.join(config.run_dir, 'model.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config.__dict__,
                'attack_names': attack_names,
                'best_thresholds': best_thresholds,
                'metrics': metrics
            }, model_path)
            print(f"💾 模型已保存: {model_path}")

            duration = (time.time() - t0) / 60
            print("\n" + "=" * 80)
            print(f"✅ FairGraphSAGE 训练完成！总耗时: {duration:.2f} 分钟")
            print(f"🏆 最终测试集 Macro-F1: {metrics['macro_f1']:.4f}")
            print("=" * 80)

            return metrics['macro_f1']
        else:
            print("❌ 评估失败")
            return 0.0

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

# ==================== 程序入口 ====================
if __name__ == "__main__":
    final_f1 = main()
    print(f"\n程序结束，Macro-F1: {final_f1:.4f}")