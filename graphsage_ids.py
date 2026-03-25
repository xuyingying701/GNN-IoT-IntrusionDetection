import os
import sys
import gc
import warnings
import time
import copy
from datetime import datetime

# ==================== 线程控制（仅限Windows CPU）====================
if sys.platform == 'win32':
    try:
        import torch
        if not torch.cuda.is_available():
            os.environ["OMP_NUM_THREADS"] = "2"  # 适度限制，非单线程
            os.environ["MKL_NUM_THREADS"] = "2"
    except:
        pass

# 非交互式绘图后端
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, f1_score, accuracy_score,
                             confusion_matrix, roc_auc_score, average_precision_score)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# 固定随机种子
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 70)
print("🚀 FairGraphSAGE - 与V6.8严格对齐的对比基线")
print("=" * 70)


# ==================== 配置（与V6.8完全一致）====================
class Config:
    # 基础配置 - 与V6.8完全一致
    random_seed = 42
    hidden_channels = 128
    num_heads = 4          # 仅用于兼容，SAGE不用
    num_layers = 2
    epochs = 300
    patience = 50
    learning_rate = 0.0005
    dropout = 0.2          # 与V6.8一致，不是0.3！
    batch_size = 10000

    # 数据路径
    data_path = '../data/train_test_network.csv'
    output_dir = './outputs_fair_sage'


# ==================== 数据加载器（复用V6.8逻辑）====================
class DataProcessor:
    """与V6.8完全一致的数据处理"""
    def __init__(self, config):
        self.config = config
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.attack_encoder = LabelEncoder()

    def load_data(self):
        print(f"加载数据: {self.config.data_path}")
        df = pd.read_csv(self.config.data_path)
        print(f"数据形状: {df.shape}")
        return df

    def preprocess(self, df):
        target_col = 'type' if 'type' in df.columns else 'label'

        # 划分数据集 (60/20/20)
        indices = np.arange(len(df))
        temp_target = df[target_col].astype(str)

        train_idx, temp_idx = train_test_split(
            indices, test_size=0.4, random_state=self.config.random_seed,
            stratify=temp_target.values
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=self.config.random_seed,
            stratify=temp_target.values[temp_idx]
        )

        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        test_df = df.iloc[test_idx].copy()

        # 特征列
        exclude_cols = ['src_ip', 'dst_ip', target_col, 'id', 'timestamp', 'Unnamed: 0']
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        # 处理类别特征
        for col in feature_cols:
            if train_df[col].dtype == 'object':
                le = LabelEncoder()
                train_values = train_df[col].fillna('Unknown').astype(str).unique()
                all_possible = list(train_values)
                if 'Unknown' not in all_possible:
                    all_possible.append('Unknown')
                le.fit(all_possible)
                self.label_encoders[col] = le

                for subset in [train_df, val_df, test_df]:
                    subset[col] = subset[col].fillna('Unknown').astype(str)
                    subset[col] = subset[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
                    subset[col] = le.transform(subset[col])
            else:
                for subset in [train_df, val_df, test_df]:
                    subset[col] = subset[col].fillna(0)

        # 标准化数值特征
        numeric_cols = [c for c in feature_cols if train_df[c].dtype != 'object']
        if numeric_cols:
            train_df[numeric_cols] = self.scaler.fit_transform(train_df[numeric_cols])
            val_df[numeric_cols] = self.scaler.transform(val_df[numeric_cols])
            test_df[numeric_cols] = self.scaler.transform(test_df[numeric_cols])

        # 编码目标列
        train_targets = train_df[target_col].astype(str).unique()
        self.attack_encoder.fit(train_targets)

        train_df['attack_type'] = self.attack_encoder.transform(train_df[target_col].astype(str))

        most_common = train_targets[0]
        for subset in [val_df, test_df]:
            targets = subset[target_col].astype(str)
            subset['attack_type'] = targets.apply(
                lambda x: self.attack_encoder.transform([x])[0] if x in self.attack_encoder.classes_
                else self.attack_encoder.transform([most_common])[0]
            )

        attack_names = {i: name for i, name in enumerate(self.attack_encoder.classes_)}

        # 合并数据集
        train_df['_split'] = 'train'
        val_df['_split'] = 'val'
        test_df['_split'] = 'test'
        final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        train_mask = (final_df['_split'] == 'train').values
        val_mask = (final_df['_split'] == 'val').values
        test_mask = (final_df['_split'] == 'test').values

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]

        # 打印分布
        print(f"\n📊 训练集类别分布:")
        train_only = final_df[train_mask]
        for i, name in attack_names.items():
            count = (train_only['attack_type'] == i).sum()
            print(f"   {name}: {count}")

        return final_df, train_idx, val_idx, test_idx, attack_names, feature_cols


# ==================== 图构建器（与V6.8完全一致）====================
class GraphBuilder:
    """与V6.8完全一致的图构建"""
    def __init__(self):
        self.ip_to_id = {}

    def build(self, df, train_idx, val_idx, test_idx, feature_cols):
        print("\n[2/7] 构建图结构...")

        unique_ips = pd.unique(pd.concat([df['src_ip'], df['dst_ip']]))
        self.ip_to_id = {ip: i for i, ip in enumerate(unique_ips)}
        num_nodes = len(unique_ips)

        src_ids = df['src_ip'].map(self.ip_to_id).values.astype(np.int32)
        dst_ids = df['dst_ip'].map(self.ip_to_id).values.astype(np.int32)

        edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long)
        edge_attr = torch.tensor(df[feature_cols].values.astype(np.float32), dtype=torch.float)
        edge_labels = torch.tensor(df['attack_type'].values, dtype=torch.long)
        num_edges = edge_index.size(1)

        train_mask = torch.zeros(num_edges, dtype=torch.bool)
        val_mask = torch.zeros(num_edges, dtype=torch.bool)
        test_mask = torch.zeros(num_edges, dtype=torch.bool)

        train_mask[torch.tensor(train_idx, dtype=torch.long)] = True
        val_mask[torch.tensor(val_idx, dtype=torch.long)] = True
        test_mask[torch.tensor(test_idx, dtype=torch.long)] = True

        # 节点特征构建
        node_degree = torch.zeros(num_nodes, dtype=torch.float32)
        node_feat_sum = torch.zeros((num_nodes, edge_attr.size(1)), dtype=torch.float32)

        batch_size = 50000  # 分批处理避免内存问题
        for i in range(0, num_edges, batch_size):
            end = min(i + batch_size, num_edges)
            batch_src = edge_index[0, i:end]
            batch_dst = edge_index[1, i:end]
            batch_attr = edge_attr[i:end]

            node_degree.index_add_(0, batch_src, torch.ones(end - i))
            node_degree.index_add_(0, batch_dst, torch.ones(end - i))
            node_feat_sum.index_add_(0, batch_src, batch_attr)
            node_feat_sum.index_add_(0, batch_dst, batch_attr)

        node_degree_safe = node_degree.clamp(min=1)
        node_feat_mean = node_feat_sum / node_degree_safe.unsqueeze(1)

        train_edge_indices = torch.where(train_mask)[0]
        global_mean = edge_attr[train_edge_indices].mean(dim=0) if len(train_edge_indices) > 0 else edge_attr.mean(dim=0)
        node_feat_mean[node_degree == 0] = global_mean

        degree_feat = torch.log1p(node_degree).unsqueeze(1)
        x = torch.cat([node_feat_mean, degree_feat], dim=1).to(torch.float32)

        # 标准化（仅基于训练集）
        if len(train_edge_indices) > 0:
            train_nodes = torch.unique(torch.cat([
                edge_index[0, train_edge_indices],
                edge_index[1, train_edge_indices]
            ]))

            if len(train_nodes) > 0:
                x_mean = x[train_nodes].mean(dim=0)
                x_std = x[train_nodes].std(dim=0) + 1e-8
                x = (x - x_mean) / x_std

                ea_mean = edge_attr[train_edge_indices].mean(dim=0)
                ea_std = edge_attr[train_edge_indices].std(dim=0) + 1e-8
                edge_attr = (edge_attr - ea_mean) / ea_std

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.y = edge_labels
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        print(f"   节点数: {num_nodes}")
        print(f"   边数: {num_edges}")
        print(f"   特征维度: {x.size(1)}")

        return data


# ==================== 公平对比的模型（无BatchNorm）====================
class FairGraphSAGE(nn.Module):
    """与V6.8架构对齐的GraphSAGE - 无BatchNorm"""
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels, out_channels, dropout=0.2):
        super().__init__()
        self.node_encoder = nn.Linear(node_in_channels, hidden_channels)
        self.edge_encoder = nn.Linear(edge_in_channels, hidden_channels)

        # 2层SAGEConv（与V6.8层数一致）
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        self.dropout = nn.Dropout(dropout)

        # 分类器维度：与V6.8完全一致
        # V6.8: Linear(hidden*3, hidden*2) -> ReLU -> Dropout -> Linear(hidden*2, out)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, out_channels)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.node_encoder(x))
        edge_feat = F.relu(self.edge_encoder(edge_attr))

        # 第一层卷积 + dropout
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)

        # 第二层卷积 + dropout
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)

        src, dst = edge_index
        edge_features = torch.cat([x[src], x[dst], edge_feat], dim=1)

        return self.classifier(edge_features)


# ==================== 类别权重（与V6.8一致）====================
def get_class_weights(y_train, attack_names, config):
    present_classes = np.unique(y_train)
    class_weights_dict = compute_class_weight('balanced', classes=present_classes, y=y_train)
    full_class_weights = np.ones(len(attack_names), dtype=np.float32)
    for i, cls in enumerate(present_classes):
        full_class_weights[cls] = class_weights_dict[i]

    # 与V6.8完全一致的类别增强
    class_specific_boost = {
        'injection': 2.2,
        'password': 2.5,
        'mitm': 1.0,
    }

    attack_name_to_id = {v: k for k, v in attack_names.items()}
    for class_name, boost in class_specific_boost.items():
        if class_name in attack_name_to_id:
            class_id = attack_name_to_id[class_name]
            full_class_weights[class_id] *= boost
            print(f"🎯 {class_name} (ID:{class_id}) 权重: {full_class_weights[class_id]:.2f}")

    print(f"\n⚖️ 最终类别权重:")
    for i, name in attack_names.items():
        print(f"   {name}: {full_class_weights[i]:.2f}")

    return torch.tensor(full_class_weights, dtype=torch.float)


# ==================== 训练器 =====================
class Trainer:
    def __init__(self, model, config, attack_names, device):
        self.model = model
        self.config = config
        self.attack_names = attack_names
        self.device = device
        self.best_val_f1 = 0
        self.best_state = None
        self.patience_counter = 0
        self.train_losses = []
        self.val_f1_scores = []

    def train_epoch(self, data, criterion, optimizer):
        self.model.train()
        optimizer.zero_grad()
        out = self.model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, data, mask_type='val'):
        self.model.eval()
        out = self.model(data)
        pred = out.argmax(dim=1)
        mask = getattr(data, f'{mask_type}_mask')
        return pred[mask].cpu(), data.y[mask].cpu()

    def train(self, data, criterion, optimizer, scheduler):
        print(f"\n开始训练 (Epochs: {self.config.epochs})...")
        print(f"{'Epoch':<6} | {'Loss':<10} | {'Val F1':<8} | {'Best Val':<8}")
        print("-" * 45)

        for epoch in range(1, self.config.epochs + 1):
            loss = self.train_epoch(data, criterion, optimizer)
            scheduler.step()

            self.train_losses.append(loss)

            # 验证
            val_pred, val_true = self.evaluate(data, 'val')
            val_f1 = f1_score(val_true.numpy(), val_pred.numpy(), average='macro', zero_division=0)
            self.val_f1_scores.append(val_f1)

            # 早停
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if epoch % 20 == 0:
                print(f"{epoch:<6} | {loss:.4f}   | {val_f1:.4f}   | {self.best_val_f1:.4f}")

            if self.patience_counter >= self.config.patience:
                print(f"⏱️ 早停于 Epoch {epoch}")
                break

        # 加载最佳模型
        if self.best_state:
            self.model.load_state_dict(self.best_state)
            print(f"\n✅ 加载最佳验证集模型 (Val F1: {self.best_val_f1:.4f})")


# ==================== 评估指标 =====================
def evaluate_model(model, data, attack_names, config):
    print("\n[最终测试集评估]")

    model.eval()
    with torch.no_grad():
        out = model(data)
        probs = F.softmax(out, dim=1)
        pred = out.argmax(dim=1)

        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred = pred[data.test_mask].cpu().numpy()
        y_prob = probs[data.test_mask].cpu().numpy()

    # 基础指标
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    print("\n" + "=" * 70)
    print("📊 测试集评估结果")
    print("=" * 70)
    print(f"🎯 Macro F1:    {macro_f1:.4f}")
    print(f"🎯 Micro F1:    {micro_f1:.4f}")
    print(f"🎯 Weighted F1: {weighted_f1:.4f}")
    print(f"📈 Accuracy:    {accuracy:.4f}")

    # AUC指标
    try:
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        y_true_bin = label_binarize(y_true, classes=unique_labels)

        auc_roc_scores = []
        auc_pr_scores = []

        for i, label in enumerate(unique_labels):
            if len(np.unique(y_true_bin[:, i])) > 1:
                auc_roc = roc_auc_score(y_true_bin[:, i], y_prob[:, label])
                auc_roc_scores.append(auc_roc)

            auc_pr = average_precision_score(y_true_bin[:, i], y_prob[:, label])
            auc_pr_scores.append(auc_pr)

        print(f"📊 Mean AUC-ROC: {np.mean(auc_roc_scores):.4f}")
        print(f"📊 Mean AUC-PR:  {np.mean(auc_pr_scores):.4f}")
    except Exception as e:
        print(f"AUC 计算跳过: {e}")

    print("=" * 70)

    # 分类报告
    unique_labels_sorted = sorted(list(set(y_true) | set(y_pred)))
    target_names = [attack_names.get(i, f'Class_{i}') for i in unique_labels_sorted]
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, labels=unique_labels_sorted,
                               target_names=target_names, digits=4))

    # 混淆矩阵
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels_sorted)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix - FairGraphSAGE")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(config.output_dir, exist_ok=True)
    cm_path = os.path.join(config.output_dir, f'confusion_matrix_{timestamp}.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()

    print(f"\n✅ 混淆矩阵已保存: {cm_path}")

    return macro_f1


# ==================== 主函数 =====================
def main():
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)

    # 数据处理
    processor = DataProcessor(config)
    builder = GraphBuilder()

    df = processor.load_data()
    df, train_idx, val_idx, test_idx, attack_names, feature_cols = processor.preprocess(df)
    data = builder.build(df, train_idx, val_idx, test_idx, feature_cols)

    # 清理内存
    del df
    gc.collect()

    # 类别权重
    y_train = data.y[data.train_mask].numpy()
    class_weights = get_class_weights(y_train, attack_names, config)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 移动数据到设备
    data = data.to(device)
    class_weights = class_weights.to(device)

    # 模型
    model = FairGraphSAGE(
        node_in_channels=data.x.size(1),
        edge_in_channels=data.edge_attr.size(1),
        hidden_channels=config.hidden_channels,
        out_channels=len(attack_names),
        dropout=config.dropout
    ).to(device)

    print(f"\n🤖 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # 训练
    trainer = Trainer(model, config, attack_names, device)
    trainer.train(data, criterion, optimizer, scheduler)

    # 评估
    final_f1 = evaluate_model(model, data, attack_names, config)

    print("\n" + "=" * 70)
    print(f"✅ 对比实验完成！Macro-F1: {final_f1:.4f}")
    print("=" * 70)

    return final_f1


if __name__ == "__main__":
    t0 = time.time()
    final_f1 = main()
    duration = (time.time() - t0) / 60
    print(f"\n⏱️ 总耗时: {duration:.2f} 分钟")