import os
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report,confusion_matrix)
import warnings
import time
import matplotlib.pyplot as plt
import seaborn as sns  # 画热力图（混淆矩阵）
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

    # 初始化组件
    processor = DataProcessor(config)  # 数据处理器：负责加载和预处理数据
    builder = GraphBuilder()  # 图构建器：将DataFrame转换为图结构

    # 加载和预处理数据
    df = processor.load_data()
    df, train_idx, val_idx, test_idx, attack_names, feature_cols = processor.preprocess(df)

    # 构建图
    data = builder.build(df, train_idx, val_idx, test_idx, feature_cols)
    #=========================
    # 在 data = builder.build(...) 之后添加：

    print("\n[验证] 检查数据泄露...")
    train_nodes = torch.unique(torch.cat([
        data.edge_index[0, data.train_mask],
        data.edge_index[1, data.train_mask]
    ]))
    test_nodes = torch.unique(torch.cat([
        data.edge_index[0, data.test_mask],
        data.edge_index[1, data.test_mask]
    ]))

    # 检查测试节点是否在训练节点中
    overlap = torch.isin(test_nodes, train_nodes).sum().item()
    total_test = len(test_nodes)
    print(f"   训练节点数: {len(train_nodes)}")
    print(f"   测试节点数: {total_test}")
    print(f"   重叠节点数: {overlap} ({overlap / total_test * 100:.1f}%)")

    if overlap == total_test:
        print("   ⚠️  警告：所有测试节点都在训练集中出现过（图结构特性，无法避免）")
        print("   ✅ 但节点特征只从训练边构建，无特征泄露")
    else:
        print(f"   ✅ 有 {total_test - overlap} 个全新测试节点")
        #===========================================================

    # 清理内存
    del df
    gc.collect()

    # 计算类别权重（使用平滑权重）
    y_train = data.y[data.train_mask].numpy()  # 提取训练集的标签
    present_classes = np.unique(y_train)  # 找出训练集中出现的所有类别
    class_weights_dict = compute_class_weight('balanced', classes=present_classes,
                                              y=y_train)  # 自动平衡权重，让少数类获得更高权重,返回一个字典或数组，每个类别对应一个权重

    # 将计算出的类别权重进行平滑处理
    full_class_weights = np.ones(len(attack_names), dtype=np.float32)
    for i, cls in enumerate(present_classes):
        full_class_weights[cls] = class_weights_dict[i] ** 0.7  # 平滑权重

    # 创建一个反向映射字典
    attack_name_to_id = {v: k for k, v in attack_names.items()}
    # 类别ID到gamma值的映射表
    class_gamma_map = {}

    # 对特定攻击类别的权重进行额外增强,让模型更加关注这些攻击类型
    for class_name, boost in config.class_specific_boost.items():
        if class_name in attack_name_to_id:
            class_id = attack_name_to_id[class_name]
            full_class_weights[class_id] *= boost  # 将该类别的权重乘以增强倍数
            print(f"🎯 {class_name} (ID:{class_id}) 权重: {full_class_weights[class_id]:.2f}")

    # 为特定的攻击类别设置focal loss的gamma参数，让不同攻击类型使用不同的聚焦程度
    for class_name, gamma in config.class_gamma.items():
        if class_name in attack_name_to_id:
            class_gamma_map[attack_name_to_id[class_name]] = gamma

    class_weights = torch.tensor(full_class_weights, dtype=torch.float)

    print(f"\n⚖️ 最终类别权重:")
    for i, name in attack_names.items():
        print(f"   {name}: {full_class_weights[i]:.2f}")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 将 数据 和 权重张量 移动到指定的计算设备（CPU）上
    data = data.to(device)
    class_weights = class_weights.to(device)

    # 创建一个图Transformer模型
    model = GraphTransformer(
        data.x.size(1),  # 节点特征维度
        data.edge_attr.size(1),  # 边特征维度
        config.hidden_channels,  # 隐藏层维度
        len(attack_names),  # 输出类别数
        config.num_heads,  # 注意力头数
        config.num_layers,  # Transformer层数
        config.dropout  # Dropout比率
    ).to(device)

    print(
        f"\n🤖 模型参数量: {sum(p.numel() for p in model.parameters()):,}")  # sum(p.numel() for p in model.parameters()):,

    # 损失函数criterion          根据配置选择是否使用标签平滑
    if config.use_label_smoothing:  # 如果配置里开启了标签平滑(模型不会太自信，泛化能力更强)
        # 使用带标签平滑的 Focal Loss
        criterion = FocalLoss(
            weight=class_weights,  # 类别权重（处理样本不平衡）
            class_gamma=class_gamma_map,  # 各类别的gamma值（处理检测难度）
            default_gamma=config.base_focal_gamma,  # 默认gamma
            label_smoothing=config.label_smoothing  # 标签平滑系数（0.1）
        )
    else:  # 如果没开启标签平滑
        # 使用不带标签平滑的 Focal Loss
        criterion = FocalLoss(
            weight=class_weights,
            class_gamma=class_gamma_map,
            default_gamma=config.base_focal_gamma,
            label_smoothing=0.0  # 平滑系数为0，等于不使用
        )

    # 创建一个 AdamW 优化器(用来更新模型的参数)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # 创建一个学习率调度器，当模型性能不再提升时，自动降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,  # 要调整哪个优化器
        mode='max',  # 监控什么（'max'=越大越好，比如准确率）
        factor=0.5,  # 学习率降低多少（乘以0.5，减半）
        patience=10,  # 等多少轮没提升才降低
        verbose=True  # 打印提示信息
    )

    # 创建一个训练器对象
    trainer = Trainer(model, config, attack_names, device)

    # 开始训练模型，并获取训练好的模型和最佳阈值
    model, best_thresholds = trainer.train(data, criterion, optimizer, scheduler)

    # 最终评估
    print("\n[4/4] 最终测试集评估...")
    val_loader = EdgeBatchLoader(data, config, shuffle=False)
    test_preds, test_labels, test_probs = trainer.evaluate(val_loader, 'test')

    if test_preds is not None:
        y_true = test_labels.numpy()
        y_pred_base = test_preds.numpy()
        y_prob = test_probs.numpy()

        # 应用自适应阈值
        if config.use_adaptive_threshold and best_thresholds is not None:
            threshold_optimizer = AdaptiveThresholdOptimizer(config.threshold_strategy)
            threshold_optimizer.thresholds = best_thresholds
            y_pred_optimized = threshold_optimizer.predict(test_probs).numpy()
        else:
            y_pred_optimized = y_pred_base

        # 计算所有指标（包括误报率）
        metrics = MetricsCalculator.calculate_all(
            y_true, y_pred_optimized, y_prob, attack_names
        )

        # ================= 添加调试代码 =================
        print("\n[调试] 混淆矩阵:")
        cm = confusion_matrix(y_true, y_pred_optimized)
        print(cm)

        # 找到正常类的索引
        normal_class_id = None
        for idx, name in attack_names.items():
            if name.lower() == 'normal':
                normal_class_id = idx
                break

        if normal_class_id is not None:
            print(f"\n[调试] 正常类 '{attack_names[normal_class_id]}' 索引: {normal_class_id}")
            print(f"[调试] 正常类的列: {cm[:, normal_class_id]}")
            print(f"[调试] 正常类的行: {cm[normal_class_id, :]}")

            fp = cm[:, normal_class_id].sum() - cm[normal_class_id, normal_class_id]
            tn = cm[normal_class_id, normal_class_id]
            print(f"[调试] FP (误报): {fp}, TN (正确): {tn}")
            print(f"[调试] 误报率 = {fp} / ({fp} + {tn}) = {fp / (fp + tn) if (fp + tn) > 0 else 0:.4f}")

            fn = cm[normal_class_id, :].sum() - cm[normal_class_id, normal_class_id]
            tp = cm[normal_class_id, normal_class_id]
            print(f"[调试] FN (漏报): {fn}, TP (正确): {tp}")
            print(f"[调试] 漏报率 = {fn} / ({fn} + {tp}) = {fn / (fn + tp) if (fn + tp) > 0 else 0:.4f}")
        # ================= 调试结束 =================

        # 打印结果
        print("\n" + "=" * 70)
        print(f"🏆 最终 Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"📊 Accuracy: {metrics['accuracy']:.4f}")
        print(f"📈 Mean AUC-ROC: {metrics.get('mean_auc_roc', 0):.4f}")
        print(f"📉 Mean AUC-PR: {metrics.get('mean_auc_pr', 0):.4f}")
        print(f"❌ 宏平均误报率 (Macro-FPR): {metrics.get('macro_fpr', 0):.4f} ({metrics.get('macro_fpr', 0):.2%})")
        print(f"⚠️  宏平均漏报率 (Macro-FNR): {metrics.get('macro_fnr', 0):.4f} ({metrics.get('macro_fnr', 0):.2%})")
        print(f"📊 二分类误报率 (Binary-FPR): {metrics.get('binary_fpr', 0):.4f}")
        print(f"📊 二分类漏报率 (Binary-FNR): {metrics.get('binary_fnr', 0):.4f}")
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
    # 混淆矩阵
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

    # 训练曲线
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
    parser.add_argument('--output_dir', type=str, default='D:/01Thesis/04代码实现\project/results',
                        help='Output directory')
    return parser.parse_args()


# ================= 主程序入口 =================
if __name__ == "__main__":
    args = parse_args()

    # 创建优化版配置
    config = Config()

    # 覆盖其他配置
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            for k, v in config_dict.items():
                if hasattr(config, k):  # 检查当前config对象是否有名为k的属性
                    setattr(config, k, v)  # 动态设置对象的属性
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