import os
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix)
import warnings
import time
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import json
import yaml
import argparse

from config import Config
from data_processor import DataProcessor
from graph_builder import GraphBuilder
from maf import MAF
from graph_transformer import GraphTransformer
from focal_loss import FocalLoss
from trainer import Trainer
from edge_batch_loader import EdgeBatchLoader
from adaptive_threshold_optimizer import AdaptiveThresholdOptimizer
from metrics_calculator import MetricsCalculator

warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = str(42)


# ================= 主训练流程 ===================
def run_optimized_training(config: Config):
    """主训练函数"""

    #初始化组件
    processor = DataProcessor(config)   #数据处理器：负责加载和预处理数据
    builder = GraphBuilder()            #图构建器：将DataFrame转换为图结构

    #加载和预处理数据
    df = processor.load_data()
    df, train_idx, val_idx, test_idx, attack_names, feature_cols = processor.preprocess(df)
    #构建图
    data = builder.build(df, train_idx, val_idx, test_idx, feature_cols)

    #清理内存
    del df
    gc.collect()

    # 计算类别权重
    y_train = data.y[data.train_mask].numpy()           #训练集标签
    present_classes = np.unique(y_train)                #出现的类别ID
    class_weights_dict = compute_class_weight('balanced', classes=present_classes, y=y_train)       #每个类别的权重

    full_class_weights = np.ones(len(attack_names), dtype=np.float32)
    for i, cls in enumerate(present_classes):
        full_class_weights[cls] = class_weights_dict[i] ** 0.7

    attack_name_to_id = {v: k for k, v in attack_names.items()}
    class_gamma_map = {}

    for class_name, boost in config.class_specific_boost.items():
        if class_name in attack_name_to_id:
            class_id = attack_name_to_id[class_name]
            full_class_weights[class_id] *= boost
            print(f"🎯 {class_name} (ID:{class_id}) 权重: {full_class_weights[class_id]:.2f}")

    for class_name, gamma in config.class_gamma.items():
        if class_name in attack_name_to_id:
            class_gamma_map[attack_name_to_id[class_name]] = gamma

    class_weights = torch.tensor(full_class_weights, dtype=torch.float)

    print(f"\n⚖️ 最终类别权重:")
    for i, name in attack_names.items():
        print(f"   {name}: {full_class_weights[i]:.2f}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    data = data.to(device)
    class_weights = class_weights.to(device)

    model = GraphTransformer(
        data.x.size(1),
        data.edge_attr.size(1),
        config.hidden_channels,
        len(attack_names),
        config.num_heads,
        config.num_layers,
        config.dropout
    ).to(device)

    print(f"\n🤖 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    if config.use_label_smoothing:
        criterion = FocalLoss(
            weight=class_weights,
            class_gamma=class_gamma_map,
            default_gamma=config.base_focal_gamma,
            label_smoothing=config.label_smoothing
        )
    else:
        criterion = FocalLoss(
            weight=class_weights,
            class_gamma=class_gamma_map,
            default_gamma=config.base_focal_gamma,
            label_smoothing=0.0
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10,
        verbose=True
    )

    trainer = Trainer(model, config, attack_names, device)

    model, best_thresholds = trainer.train(data, criterion, optimizer, scheduler)

    # 最终评估
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

        # 计算所有指标（MetricsCalculator 内部会打印详细指标）
        metrics = MetricsCalculator.calculate_all(
            y_true, y_pred_optimized, y_prob, attack_names
        )

        #打印结果
        print("\n" + "=" * 70)
        print(f"🏆 最终 Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"📊 Accuracy: {metrics['accuracy']:.4f}")
        print(f"📈 Mean AUC-ROC: {metrics.get('mean_auc_roc', 0):.4f}")
        print(f"📉 Mean AUC-PR: {metrics.get('mean_auc_pr', 0):.4f}")
        print(f"❌ 宏平均误报率 (Macro-FPR): {metrics.get('macro_fpr', 0):.4f}")
        print(f"⚠️ 宏平均漏报率 (Macro-FNR): {metrics.get('macro_fnr', 0):.4f}")
        print(f"⚖️ 加权误报率: {metrics['weighted_fpr']:.4f}")
        print(f"⚖️ 加权漏报率: {metrics['weighted_fnr']:.4f}")
        print("=" * 70)

        # 分类报告
        unique_labels = sorted(list(set(y_true) | set(y_pred_optimized)))
        target_names = [attack_names.get(i, f'Class_{i}') for i in unique_labels]
        print("\n分类报告:")
        print(classification_report(y_true, y_pred_optimized, labels=unique_labels,
                                    target_names=target_names, digits=4))

        # 保存结果
        results = {
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

        # 绘制混淆矩阵和训练曲线
        plot_results(y_true, y_pred_optimized, target_names, trainer, config.run_dir)

        return metrics['macro_f1']
    else:
        print("❌ 评估失败")
        return 0.0


def plot_results(y_true, y_pred, target_names, trainer, save_dir):
    """绘制结果"""
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix (Counts)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix (Normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    if trainer.train_losses:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(trainer.train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        epochs = range(0, len(trainer.val_f1s) * 5, 5)
        plt.plot(epochs, trainer.val_f1s, label='Val F1')
        plt.plot(epochs, trainer.test_f1s, label='Test F1')
        plt.title('F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('Macro-F1')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Graph Transformer for IoT Intrusion Detection')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden channels')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='D:/01Thesis/04代码实现/project/results',
                        help='Output directory')
    return parser.parse_args()


# ================= 主程序入口 =================
if __name__ == "__main__":
    args = parse_args()

    #创建优化版配置
    config = Config()

    #覆盖其他配置
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            for k, v in config_dict.items():
                if hasattr(config, k):          #检查当前config对象是否有名为k的属性
                    setattr(config, k, v)       #动态设置对象的属性
    if args.hidden_channels:
        config.hidden_channels = args.hidden_channels
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.output_dir:
        config.output_dir = args.output_dir

    # 重新初始化运行目录
    config.__post_init__()

    # 保存配置
    config.save()

    print("=" * 70)
    print(f"🚀 图Transformer - 物联网入侵检测")
    print("=" * 70)
    print(f"数据路径: {config.data_path}")
    print(f"输出目录: {config.run_dir}")
    print(f"隐藏层维度: {config.hidden_channels}")
    print(f"学习率: {config.learning_rate}")
    print(f"Dropout: {config.dropout}")
    print(f"标签平滑: {config.label_smoothing}")
    print("=" * 70)

    try:
        t0 = time.time()
        final_f1 = run_optimized_training(config)
        duration = (time.time() - t0) / 60

        print("\n" + "=" * 70)
        print(f"✅ 训练完成！总耗时: {duration:.2f} 分钟")
        print(f"🏆 最终测试集 Macro-F1: {final_f1:.4f}")

        print("=" * 70)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()