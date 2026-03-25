import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import os
import sys
import warnings
import time
import traceback  # ✅ 修复: 添加缺失的导入
from datetime import datetime

warnings.filterwarnings("ignore")

# ==========================================
# 配置部分 (可根据数据集调整)
# ==========================================

# 【路径配置】
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'train_test_network.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'rf_baseline')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 【实验配置】
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2  # 验证集占训练集的比例

# 【关键开关】时间序列表划分
USE_TEMPORAL_SPLIT = False
TIME_COLUMN = 'timestamp'

# 【特征配置】硬编码特征列表 (针对ToN-IoT优化)
NUMERIC_FEATURES = [
    'duration', 'src_bytes', 'dst_bytes', 'src_pkts', 'dst_pkts', 'missed_bytes'
]
CATEGORICAL_FEATURES = [
    'proto', 'service', 'ssl_version', 'http_method', 'conn_state'
]

TARGET_COLUMN = 'type'
NORMAL_CLASS_NAME = 'normal'  # ✅ 已修正: ToN-IoT数据集中为小写

# 【模型配置】
N_ESTIMATORS = 200
MAX_DEPTH = 30  # 限制深度防止过拟合
CLASS_WEIGHT = 'balanced'


# ==========================================
# 工具函数
# ==========================================

def save_results(results_dict, y_test, y_pred, feature_importance, class_names, timestamp):
    """
    保存所有结果到文件
    """
    run_dir = os.path.join(OUTPUT_DIR, f"rf_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n💾 保存结果到: {run_dir}")

    # 1. 保存指标 (JSON)
    with open(os.path.join(run_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    # 2. 保存分类报告
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    with open(os.path.join(run_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Random Forest Baseline - {timestamp}\n")
        f.write("="*70 + "\n\n")
        f.write(report)
        # ✅ 修复: 使用正确的字典键名 'metrics' -> 'fpr'
        f.write(f"\n\nFPR (False Positive Rate): {results_dict['metrics']['fpr']:.4f}\n")
        f.write(f"FNR (False Negative Rate): {results_dict['metrics']['fnr']:.4f}\n")

    # 3. 保存特征重要性
    feature_importance.to_csv(os.path.join(run_dir, 'feature_importance.csv'), index=False)

    # 4. 保存混淆矩阵数据
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(run_dir, 'confusion_matrix.csv'))

    # 5. 绘制混淆矩阵图
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Counts)')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.subplot(1, 2, 2)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Normalized)')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 6. 绘制特征重要性图 (Top 15)
    plt.figure(figsize=(10, 8))
    top_n = min(15, len(feature_importance))
    sns.barplot(data=feature_importance.head(top_n), y='Feature', x='Importance', palette='viridis')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'feature_importance.png'), dpi=150)
    plt.close()

    print(f"   ✓ metrics.json")
    print(f"   ✓ classification_report.txt")
    print(f"   ✓ feature_importance.csv/png")
    print(f"   ✓ confusion_matrix.csv/png")

    return run_dir


# ==========================================
# 主程序
# ==========================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*70}")
    print("🌲 优化版随机森林基线模型")
    print(f"{'='*70}")
    print(f"数据路径: {DATA_PATH}")
    print(f"时间划分: {'启用' if USE_TEMPORAL_SPLIT else '禁用'}")
    print(f"随机种子: {RANDOM_STATE}")
    print(f"{'='*70}\n")

    try:
        # ---------- 1. 加载数据 ----------
        print(f"📂 [1/5] 加载数据...")
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"找不到文件: {DATA_PATH}")

        df = pd.read_csv(DATA_PATH)
        print(f"   原始数据: {df.shape[0]} 行 × {df.shape[1]} 列")

        # 检查目标列
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"目标列 '{TARGET_COLUMN}' 不存在!")

        # 显示类别分布
        print(f"\n   类别分布:")
        target_counts = df[TARGET_COLUMN].value_counts()
        for cls, count in target_counts.items():
            print(f"     {cls}: {count} ({count/len(df)*100:.2f}%)")

        # ---------- 2. 预处理 ----------
        print(f"\n🔧 [2/5] 特征预处理...")

        # 标签编码
        le_target = LabelEncoder()
        y = le_target.fit_transform(df[TARGET_COLUMN])
        class_names = le_target.classes_
        n_classes = len(class_names)

        # 找到normal类索引 (关键!)
        if NORMAL_CLASS_NAME in class_names:
            normal_idx = list(class_names).index(NORMAL_CLASS_NAME)
            print(f"   🛡️ 正常类: '{NORMAL_CLASS_NAME}' (索引 {normal_idx})")
        else:
            # 尝试常见变体
            alternatives = ['normal', 'Normal', 'NORMAL', 'benign', 'Benign', 'BENIGN']
            normal_idx = None
            for alt in alternatives:
                if alt in class_names:
                    normal_idx = list(class_names).index(alt)
                    print(f"   🛡️ 正常类: '{alt}' (索引 {normal_idx}) [自动识别]")
                    break
            if normal_idx is None:
                print(f"   ⚠️ 警告: 未找到标准正常类名, 假设索引0为正常类")
                normal_idx = 0

        # 特征处理
        df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0)
        df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].fillna('Unknown')

        selected_features = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES
                            if c in df.columns]

        X = df[selected_features].copy()

        # 类别特征编码
        for col in CATEGORICAL_FEATURES:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        print(f"   特征数: {len(selected_features)} (数值{len(NUMERIC_FEATURES)}, 类别{len(CATEGORICAL_FEATURES)})")

        # ---------- 3. 数据划分 ----------
        print(f"\n✂️ [3/5] 数据划分...")

        if USE_TEMPORAL_SPLIT and TIME_COLUMN in df.columns:
            # 时间序列表划分
            print(f"   ⏳ 时间序列表划分 (按 {TIME_COLUMN})")
            df_sorted = df.sort_values(by=TIME_COLUMN).reset_index(drop=True)
            X_sorted = X.loc[df_sorted.index].values
            y_sorted = y[df_sorted.index]

            # 划分点: 训练60% / 验证20% / 测试20%
            n = len(df_sorted)
            train_end = int(n * (1 - TEST_SIZE - VAL_SIZE))
            val_end = int(n * (1 - TEST_SIZE))

            X_train, X_val, X_test = (
                X_sorted[:train_end],
                X_sorted[train_end:val_end],
                X_sorted[val_end:]
            )
            y_train, y_val, y_test = (
                y_sorted[:train_end],
                y_sorted[train_end:val_end],
                y_sorted[val_end:]
            )
        else:
            # 随机分层划分
            if USE_TEMPORAL_SPLIT:
                print(f"   ⚠️ 未找到时间列 '{TIME_COLUMN}', 回退到随机划分")
            print(f"   🔀 随机分层划分")

            # 先划分测试集
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
            )
            # 再划分验证集
            val_ratio = VAL_SIZE / (1 - TEST_SIZE)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_temp
            )

        print(f"   训练集: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   验证集: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
        print(f"   测试集: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

        # ---------- 4. 训练模型 ----------
        print(f"\n🚀 [4/5] 训练随机森林...")
        print(f"   参数: n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, class_weight={CLASS_WEIGHT}")

        start_time = time.time()
        rf_model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            class_weight=CLASS_WEIGHT,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )
        rf_model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # 验证集评估
        y_val_pred = rf_model.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred, average='macro')

        print(f"   ✅ 训练完成! 耗时: {train_time:.2f}s, 验证集F1: {val_f1:.4f}")

        # ---------- 5. 测试评估 ----------
        print(f"\n📊 [5/5] 测试集评估...")

        # 推理时间测试
        n_infer = min(1000, len(X_test))
        infer_start = time.time()
        for _ in range(10):
            rf_model.predict(X_test[:n_infer])
        infer_time = (time.time() - infer_start) / (10 * n_infer) * 1000

        # 完整预测
        y_pred = rf_model.predict(X_test)

        # 基础指标
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        macro_p = precision_score(y_test, y_pred, average='macro', zero_division=0)
        macro_r = recall_score(y_test, y_pred, average='macro', zero_division=0)

        # 混淆矩阵 & FPR/FNR
        cm = confusion_matrix(y_test, y_pred)
        tn = cm[normal_idx, normal_idx]
        fp = cm[normal_idx, :].sum() - tn
        fn = cm[:, normal_idx].sum() - tn
        tp = cm.sum() - tn - fp - fn

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        # 打印结果
        print(f"\n{'='*70}")
        print("📈 评估结果 (随机森林基线)")
        print(f"{'='*70}")
        print(f"Accuracy        : {acc:.4f}")
        print(f"Macro-F1        : {macro_f1:.4f}")
        print(f"Macro-Precision : {macro_p:.4f}")
        print(f"Macro-Recall    : {macro_r:.4f}")
        print(f"{'='*70}")
        print(f"❌ FPR (误报率)  : {fpr:.4f} (正常→攻击)")
        print(f"⚠️  FNR (漏报率)  : {fnr:.4f} (攻击→正常)")
        print(f"{'='*70}")
        print(f"⏱️  推理延迟      : {infer_time:.3f} ms/样本")
        print(f"🕐 训练时间      : {train_time:.2f} 秒")
        print(f"{'='*70}")

        # 每类详细指标
        print(f"\n📋 各类别性能:")
        for i, name in enumerate(class_names):
            mask = y_test == i
            if mask.sum() == 0:
                continue
            p = precision_score(y_test == i, y_pred == i, zero_division=0)
            r = recall_score(y_test == i, y_pred == i, zero_division=0)
            f1 = f1_score(y_test == i, y_pred == i, zero_division=0)
            print(f"  {name:12s}: P={p:.4f}, R={r:.4f}, F1={f1:.4f}, n={mask.sum()}")

        # 特征重要性
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print(f"\n🏆 Top 5 特征:")
        for _, row in importance_df.head(5).iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")

        # ---------- 6. 保存结果 ----------
        results = {
            'timestamp': timestamp,
            'model': 'RandomForest',
            'configuration': {
                'n_estimators': N_ESTIMATORS,
                'max_depth': MAX_DEPTH,
                'class_weight': CLASS_WEIGHT,
                'random_state': RANDOM_STATE,
                'use_temporal_split': USE_TEMPORAL_SPLIT
            },
            'metrics': {
                'accuracy': float(acc),
                'macro_f1': float(macro_f1),
                'macro_precision': float(macro_p),
                'macro_recall': float(macro_r),
                'fpr': float(fpr),
                'fnr': float(fnr),
                'train_time_sec': float(train_time),
                'inference_time_ms': float(infer_time)
            },
            'class_names': list(class_names),
            'normal_class_index': int(normal_idx),
            'feature_names': selected_features
        }

        run_dir = save_results(results, y_test, y_pred, importance_df, class_names, timestamp)

        # ---------- 7. 结论 ----------
        print(f"\n{'='*70}")
        print("💡 基线结论")
        print(f"{'='*70}")
        print(f"强基线性能: Macro-F1={macro_f1:.4f}, FPR={fpr:.4f}")

        # 找出最弱类别
        weakest = None
        weakest_f1 = 1.0
        for i, name in enumerate(class_names):
            mask = y_test == i
            if mask.sum() > 0:
                f1 = f1_score(y_test == i, y_pred == i, zero_division=0)
                if f1 < weakest_f1:
                    weakest_f1 = f1
                    weakest = name

        if weakest and weakest_f1 < 0.9:
            print(f"改进机会: {weakest}类别F1={weakest_f1:.4f} (样本极少或需拓扑特征)")

        print(f"\n下一步: 运行GNN模型,利用设备关系提升{'协同攻击检测' if weakest else '整体性能'}")
        print(f"结果保存: {run_dir}")
        print(f"{'='*70}")

        # 保存模型
        model_path = os.path.join(run_dir, 'model.pkl')
        joblib.dump({
            'model': rf_model,
            'label_encoder': le_target,
            'config': {
                'normal_idx': normal_idx,
                'class_names': list(class_names),
                'features': selected_features
            }
        }, model_path)
        print(f"\n💾 模型已保存: {model_path}")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()