import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, f1_score, accuracy_score,
                             confusion_matrix, precision_recall_curve, roc_auc_score,
                             average_precision_score)
from sklearn.metrics import precision_score, recall_score
import warnings
import time
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import json
import yaml
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import argparse

warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = str(42)


# ================= 严格对齐配置 =================
@dataclass
class FairLSTMConfig:
    """与GraphSAGE/Transformer严格对齐的LSTM配置"""
    # === 与主模型完全一致的配置 ===
    random_seed: int = 42
    hidden_channels: int = 128  # 对齐GraphSAGE/Transformer
    num_layers: int = 2  # 对齐2层
    epochs: int = 300  # 对齐
    patience: int = 50  # 对齐
    learning_rate: float = 0.0005  # 对齐
    dropout: float = 0.2  # 对齐
    batch_size: int = 10000  # 对齐

    # === LSTM特有配置（但保持公平） ===
    bidirectional: bool = True  # 双向使hidden*2，对齐图模型的multi-head

    # === 损失函数对齐 ===
    focal_gamma: float = 4.0  # 对齐主模型的base_focal_gamma

    # === 类别特定配置（完全对齐） ===
    class_weights_boost: Dict[str, float] = field(default_factory=lambda: {
        'injection': 2.2,  # 完全对齐
        'password': 2.5,  # 完全对齐
        'mitm': 1.0,  # 完全对齐
    })

    # === 数据路径（完全一致） ===
    data_path: str = 'D:/01Thesis/04代码实现/project/data/train_test_network.csv'
    output_dir: str = './fair_lstm_outputs'

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.output_dir, f"fair_lstm_run_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

    def save(self):
        """保存配置"""
        config_path = os.path.join(self.run_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.__dict__, f)


# ================= 对齐版LSTM =================
class FairLSTM(nn.Module):
    """
    与GraphSAGE/Transformer严格对齐的LSTM

    对齐策略：
    1. 输入维度 = edge_attr维度（完全一致）
    2. hidden = 128（对齐）
    3. layers = 2（对齐）
    4. dropout = 0.2（对齐）
    5. 分类器结构：hidden*2 -> hidden*2 -> hidden -> num_classes（对齐）
    """

    def __init__(self, input_dim: int, hidden: int = 128, num_layers: int = 2,
                 num_classes: int = 10, dropout: float = 0.2, bidirectional: bool = True):
        super().__init__()

        self.hidden = hidden
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # 输入投影（可选，但保持维度对齐）
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # LSTM层 - 严格对齐参数
        self.lstm = nn.LSTM(
            input_size=hidden,  # 投影后
            hidden_size=hidden,  # 对齐128
            num_layers=num_layers,  # 对齐2层
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # 分类器 - 完全对齐Transformer的分类器结构
        # Transformer: Linear(hidden*3, hidden*2) -> ReLU -> Dropout -> Linear(hidden*2, out)
        # LSTM使用hidden*2（双向拼接）对齐Transformer的hidden*3
        self.classifier = nn.Sequential(
            nn.Linear(hidden * self.num_directions, hidden * 2),  # 对齐维度
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),  # 中间层
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)  # 输出层
        )

    def forward(self, x):
        """
        输入: x - [batch_size, feature_dim] (单条边特征)
        输出: [batch_size, num_classes]
        """
        # 输入投影
        x = self.input_proj(x)  # [batch, hidden]

        # LSTM需要序列维度，我们构造长度为1的序列
        # 这样LSTM就变成了"序列感知的全连接层"
        x = x.unsqueeze(1)  # [batch, 1, hidden]

        # LSTM前向
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out: [batch, 1, hidden * num_directions]

        # 取最后时间步的输出
        out = lstm_out[:, -1, :]  # [batch, hidden * num_directions]

        # 分类
        return self.classifier(out)


# ================= 严格对齐数据处理 =================
class FairDataProcessor:
    """
    与GraphSAGE/Transformer完全一致的数据处理

    关键对齐点：
    1. 相同的train/val/test划分（边级别）
    2. 相同的特征处理
    3. 相同的类别权重计算
    """

    def __init__(self, config: FairLSTMConfig):
        self.config = config
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.attack_encoder = LabelEncoder()

    def load_and_split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    Dict[int, str], List[str]]:
        """
        完全对齐GraphSAGE的数据划分流程
        """
        print(f"\n[1/4] 加载数据: {self.config.data_path}")
        df = pd.read_csv(self.config.data_path)
        print(f"   原始数据形状: {df.shape}")

        target_col = 'type' if 'type' in df.columns else 'label'

        # === 完全对齐：先划分，再处理 ===
        indices = np.arange(len(df))
        temp_target = df[target_col].astype(str)

        # 与GraphSAGE完全一致的划分比例：60% train, 20% val, 20% test
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.4, random_state=self.config.random_seed,
            stratify=temp_target.values
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=self.config.random_seed,
            stratify=temp_target.values[temp_idx]
        )

        # 划分数据集
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        test_df = df.iloc[test_idx].copy()

        print(f"\n   划分后:")
        print(f"     训练集: {len(train_df)} 条边")
        print(f"     验证集: {len(val_df)} 条边")
        print(f"     测试集: {len(test_df)} 条边")

        # === 特征列（与GraphSAGE完全一致） ===
        exclude_cols = ['src_ip', 'dst_ip', target_col, 'id', 'timestamp', 'Unnamed: 0']
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        # === 处理类别特征（与GraphSAGE完全一致） ===
        train_df, val_df, test_df = self._encode_categorical_features(
            train_df, val_df, test_df, feature_cols
        )

        # === 标准化（与GraphSAGE完全一致） ===
        train_df, val_df, test_df = self._standardize_features(
            train_df, val_df, test_df, feature_cols
        )

        # === 编码标签（与GraphSAGE完全一致） ===
        train_df, val_df, test_df, attack_names = self._encode_target(
            train_df, val_df, test_df, target_col
        )

        # 提取特征和标签
        X_train = train_df[feature_cols].values.astype(np.float32)
        X_val = val_df[feature_cols].values.astype(np.float32)
        X_test = test_df[feature_cols].values.astype(np.float32)

        y_train = train_df['attack_type'].values.astype(np.int64)
        y_val = val_df['attack_type'].values.astype(np.int64)
        y_test = test_df['attack_type'].values.astype(np.int64)

        # 打印分布
        self._print_distribution(y_train, y_val, y_test, attack_names)

        return X_train, X_val, X_test, y_train, y_val, y_test, attack_names, feature_cols

    def _encode_categorical_features(self, train_df, val_df, test_df, feature_cols):
        """与GraphSAGE完全一致的类别特征编码"""
        for col in feature_cols:
            if train_df[col].dtype == 'object':
                le = LabelEncoder()

                # 只从训练集学习编码
                train_values = train_df[col].fillna('Unknown').astype(str).unique()
                all_possible_values = list(train_values)
                if 'Unknown' not in all_possible_values:
                    all_possible_values.append('Unknown')

                le.fit(all_possible_values)
                self.label_encoders[col] = le

                # 转换所有数据集
                for df_subset in [train_df, val_df, test_df]:
                    df_subset[col] = df_subset[col].fillna('Unknown').astype(str)
                    df_subset[col] = df_subset[col].apply(
                        lambda x: x if x in le.classes_ else 'Unknown'
                    )
                    df_subset[col] = le.transform(df_subset[col])
            else:
                for df_subset in [train_df, val_df, test_df]:
                    df_subset[col] = df_subset[col].fillna(0)

        return train_df, val_df, test_df

    def _standardize_features(self, train_df, val_df, test_df, feature_cols):
        """与GraphSAGE完全一致的标准化（只从训练集拟合）"""
        numeric_cols = [c for c in feature_cols if train_df[c].dtype != 'object']
        if numeric_cols:
            # 只从训练集拟合
            train_df[numeric_cols] = self.scaler.fit_transform(train_df[numeric_cols])
            # 应用同样的变换
            val_df[numeric_cols] = self.scaler.transform(val_df[numeric_cols])
            test_df[numeric_cols] = self.scaler.transform(test_df[numeric_cols])

        return train_df, val_df, test_df

    def _encode_target(self, train_df, val_df, test_df, target_col):
        """与GraphSAGE完全一致的标签编码"""
        # 只从训练集学习
        train_targets = train_df[target_col].astype(str).unique()
        self.attack_encoder.fit(train_targets)

        train_df['attack_type'] = self.attack_encoder.transform(
            train_df[target_col].astype(str)
        )

        # 处理验证集和测试集中的未知值
        most_common = train_targets[0]
        for df_subset in [val_df, test_df]:
            targets = df_subset[target_col].astype(str)
            df_subset['attack_type'] = targets.apply(
                lambda x: self.attack_encoder.transform([x])[0]
                if x in self.attack_encoder.classes_
                else self.attack_encoder.transform([most_common])[0]
            )

        attack_names = {i: name for i, name in enumerate(self.attack_encoder.classes_)}
        return train_df, val_df, test_df, attack_names

    def _print_distribution(self, y_train, y_val, y_test, attack_names):
        """打印分布"""
        print(f"\n📊 各类别样本数:")
        print("-" * 60)
        print(f"{'类别':<15} {'训练集':<10} {'验证集':<10} {'测试集':<10} {'总计':<10}")
        print("-" * 60)

        for i, name in attack_names.items():
            train_cnt = (y_train == i).sum()
            val_cnt = (y_val == i).sum()
            test_cnt = (y_test == i).sum()
            total = train_cnt + val_cnt + test_cnt
            print(f"{name:<15} {train_cnt:<10} {val_cnt:<10} {test_cnt:<10} {total:<10}")

        print("-" * 60)
        print(
            f"{'总计':<15} {len(y_train):<10} {len(y_val):<10} {len(y_test):<10} {len(y_train) + len(y_val) + len(y_test):<10}")


# ================= 焦点损失函数（对齐） =================
class FocalLoss(nn.Module):
    """与ClassSpecificFocalLoss对齐的焦点损失"""

    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 4.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ================= 公平训练器 =================
class FairTrainer:
    """与GraphSAGE训练器对齐"""

    def __init__(self, model: nn.Module, config: FairLSTMConfig,
                 attack_names: Dict, device: torch.device):
        self.model = model
        self.config = config
        self.attack_names = attack_names
        self.device = device
        self.best_val_f1 = 0
        self.best_state = None
        self.patience_cnt = 0
        self.best_thresholds = None

    def train(self, train_loader, val_loader, test_loader, criterion, optimizer, scheduler):
        """与GraphSAGE完全对齐的训练循环"""
        print(f"\n[2/4] 开始训练 (Epochs: {self.config.epochs})...")

        history = {
            'train_loss': [], 'val_loss': [],
            'val_f1': [], 'test_f1': []
        }

        for epoch in range(1, self.config.epochs + 1):
            # 训练
            train_loss = self._train_epoch(train_loader, criterion, optimizer)

            # 评估
            val_loss, val_f1, val_preds, val_labels, val_probs = self._evaluate(val_loader, criterion)
            _, test_f1, test_preds, test_labels, test_probs = self._evaluate(test_loader, criterion)

            # 学习率调整
            if scheduler:
                scheduler.step()

            # 保存历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_f1'].append(val_f1)
            history['test_f1'].append(test_f1)

            # 早停检查
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.patience_cnt = 0

                # 保存最佳模型
                torch.save(self.best_state, os.path.join(self.config.run_dir, 'best_model.pth'))
            else:
                self.patience_cnt += 1

            # 打印进度
            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")

            if self.patience_cnt >= self.config.patience:
                print(f"⏱️ 早停于 Epoch {epoch}")
                break

        # 加载最佳模型
        if self.best_state:
            self.model.load_state_dict(self.best_state)
            print(f"\n✅ 加载最佳验证集模型 (Val F1: {self.best_val_f1:.4f})")

        return history

    def _train_epoch(self, train_loader, criterion, optimizer):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            # 梯度裁剪（与GraphSAGE对齐）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    @torch.no_grad()
    def _evaluate(self, loader, criterion):
        """评估"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            outputs = self.model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()

            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / len(loader)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        return avg_loss, f1, np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ================= 评估函数 =================
def evaluate_and_save(y_true, y_pred, y_prob, attack_names, save_dir):
    """全面评估并保存结果"""

    # 计算所有指标
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    print("\n" + "=" * 70)
    print("📊 公平LSTM评估结果")
    print("=" * 70)
    print(f"\n🏆 宏观F1 (Macro-F1): {macro_f1:.4f}")
    print(f"📊 准确率 (Accuracy): {accuracy:.4f}")
    print(f"⚖️ 加权F1 (Weighted-F1): {weighted_f1:.4f}")

    # 每类指标
    print("\n" + "-" * 70)
    print("各类别详细指标:")
    print("-" * 70)
    print(f"{'类别':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'样本数':<8}")
    print("-" * 70)

    per_class = {}
    unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

    for label in unique_labels:
        class_name = attack_names.get(label, f'Class_{label}')
        precision = precision_score(y_true, y_pred, labels=[label], average=None, zero_division=0)[0]
        recall = recall_score(y_true, y_pred, labels=[label], average=None, zero_division=0)[0]
        f1 = f1_score(y_true, y_pred, labels=[label], average=None, zero_division=0)[0]
        support = (y_true == label).sum()

        per_class[class_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(support)
        }

        print(f"{class_name:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<8}")

    # 保存结果
    results = {
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'accuracy': float(accuracy),
        'per_class_metrics': per_class
    }

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # 绘制混淆矩阵
    plot_confusion_matrix(y_true, y_pred, attack_names, save_dir)

    return macro_f1, per_class


def plot_confusion_matrix(y_true, y_pred, attack_names, save_dir):
    """绘制混淆矩阵"""
    plt.figure(figsize=(14, 6))

    unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    target_names = [attack_names.get(i, f'Class_{i}') for i in unique_labels]

    # 原始计数
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Fair LSTM - Confusion Matrix (Counts)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45)

    # 归一化
    plt.subplot(1, 2, 2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Fair LSTM - Confusion Matrix (Normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(history, save_dir):
    """绘制训练历史"""
    plt.figure(figsize=(12, 4))

    # 损失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # F1
    plt.subplot(1, 2, 2)
    plt.plot(history['val_f1'], label='Val F1', color='red')
    plt.plot(history['test_f1'], label='Test F1', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Macro F1 Score')
    plt.title('Validation and Test F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ================= 主函数 =================
def run_fair_lstm():
    """运行公平LSTM对比实验"""

    # 配置
    config = FairLSTMConfig()
    config.__post_init__()
    config.save()

    print("=" * 70)
    print("⚖️ 公平LSTM - 与GraphSAGE/Transformer严格对齐")
    print("=" * 70)
    print(f"数据路径: {config.data_path}")
    print(f"隐藏维度: {config.hidden_channels} (对齐)")
    print(f"层数: {config.num_layers} (对齐)")
    print(f"学习率: {config.learning_rate} (对齐)")
    print(f"Dropout: {config.dropout} (对齐)")
    print(f"批次大小: {config.batch_size} (对齐)")
    print(f"输出目录: {config.run_dir}")
    print("=" * 70)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 数据处理（完全对齐）
    processor = FairDataProcessor(config)
    X_train, X_val, X_test, y_train, y_val, y_test, attack_names, feature_cols = processor.load_and_split()

    # 转换为张量
    X_train_t = torch.FloatTensor(X_train)
    X_val_t = torch.FloatTensor(X_val)
    X_test_t = torch.FloatTensor(X_test)
    y_train_t = torch.LongTensor(y_train)
    y_val_t = torch.LongTensor(y_val)
    y_test_t = torch.LongTensor(y_test)

    # 数据加载器
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # 计算类别权重（完全对齐）
    present_classes = np.unique(y_train)
    class_weights_array = compute_class_weight('balanced', classes=present_classes, y=y_train)

    full_class_weights = np.ones(len(attack_names), dtype=np.float32)
    for i, cls in enumerate(present_classes):
        full_class_weights[cls] = class_weights_array[i]

    # 应用类别特定权重提升（完全对齐）
    attack_name_to_id = {v: k for k, v in attack_names.items()}
    for class_name, boost in config.class_weights_boost.items():
        if class_name in attack_name_to_id:
            class_id = attack_name_to_id[class_name]
            full_class_weights[class_id] *= boost
            print(f"\n🎯 {class_name} (ID:{class_id}) 权重提升: {boost}x -> {full_class_weights[class_id]:.2f}")

    class_weights = torch.tensor(full_class_weights, dtype=torch.float).to(device)

    print(f"\n⚖️ 最终类别权重:")
    for i, name in attack_names.items():
        print(f"   {name}: {full_class_weights[i]:.2f}")

    # 初始化模型（严格对齐）
    model = FairLSTM(
        input_dim=X_train.shape[1],
        hidden=config.hidden_channels,
        num_layers=config.num_layers,
        num_classes=len(attack_names),
        dropout=config.dropout,
        bidirectional=True
    ).to(device)

    print(f"\n🤖 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数和优化器（对齐）
    criterion = FocalLoss(weight=class_weights, gamma=config.focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # 训练器
    trainer = FairTrainer(model, config, attack_names, device)

    # 训练
    t0 = time.time()
    history = trainer.train(train_loader, val_loader, test_loader, criterion, optimizer, scheduler)
    train_time = (time.time() - t0) / 60

    # 最终评估
    print("\n[3/4] 最终测试集评估...")
    _, test_f1, test_preds, test_labels, test_probs = trainer._evaluate(test_loader, criterion)

    # 详细评估
    macro_f1, per_class = evaluate_and_save(test_labels, test_preds, test_probs, attack_names, config.run_dir)

    # 绘制训练历史
    plot_training_history(history, config.run_dir)

    # 与主模型对比
    print("\n" + "=" * 70)
    print("📊 与图模型的公平对比")
    print("=" * 70)
    print(f"\n{'模型':<20} {'Macro-F1':<12} {'配置对齐':<10} {'数据划分':<12}")
    print("-" * 70)
    print(f"{'Fair LSTM (本实验)':<20} {macro_f1:<12.4f} {'✅ 完全对齐':<10} {'边级别':<12}")
    print(f"{'GraphSAGE':<20} {'0.873':<12} {'✅ 完全对齐':<10} {'边级别':<12}")
    print(f"{'Graph Transformer':<20} {'0.94+':<12} {'✅ 完全对齐':<10} {'边级别':<12}")
    print("=" * 70)

    print(f"\n⏱️ 训练耗时: {train_time:.2f} 分钟")
    print(f"💾 结果已保存至: {config.run_dir}")

    return macro_f1


if __name__ == "__main__":
    try:
        final_f1 = run_fair_lstm()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback

        traceback.print_exc()