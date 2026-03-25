import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import warnings
from typing import List

warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = str(42)


class GraphBuilder:
    """修复版图构建器 - 只用训练边构建节点特征，防止数据泄露"""

    def __init__(self):
        self.ip_to_id = {}

    def build(self, df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray,
              test_idx: np.ndarray, feature_cols: List[str]) -> Data:
        """构建图数据 - 修复数据泄露问题"""
        print("\n[2/4] 构建图结构...")

        # 1. 创建节点映射
        unique_ips = pd.unique(pd.concat([df['src_ip'], df['dst_ip']]))
        self.ip_to_id = {ip: i for i, ip in enumerate(unique_ips)}
        num_nodes = len(unique_ips)
        print(f"   节点数: {num_nodes}")

        # 2. 构建边索引和边特征（全量数据，用于图结构）
        src_ids = df['src_ip'].map(self.ip_to_id).values.astype(np.int64)
        dst_ids = df['dst_ip'].map(self.ip_to_id).values.astype(np.int64)
        edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long)
        edge_attr = torch.tensor(df[feature_cols].values.astype(np.float32), dtype=torch.float)
        edge_labels = torch.tensor(df['attack_type'].values, dtype=torch.long)
        num_edges = edge_index.size(1)
        print(f"   边数: {num_edges}")

        # 3. 创建掩码
        train_mask = torch.zeros(num_edges, dtype=torch.bool)
        val_mask = torch.zeros(num_edges, dtype=torch.bool)
        test_mask = torch.zeros(num_edges, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        # ===== 关键修复：只用训练边构建节点特征 =====
        print("   🛡️ 使用训练边构建节点特征（防止数据泄露）...")

        # 提取训练边
        train_edges_idx = torch.where(train_mask)[0]
        train_edge_index = edge_index[:, train_edges_idx]
        train_edge_attr = edge_attr[train_edges_idx]

        # 只用训练边构建节点特征
        node_attr = self._build_node_features_from_edges(
            train_edge_index, train_edge_attr, num_nodes
        )

        # 获取参与训练的节点（用于标准化）
        train_nodes = torch.unique(torch.cat([
            train_edge_index[0], train_edge_index[1]
        ]))

        # 标准化节点特征（只用训练节点）
        node_attr = self._normalize_features_safe(node_attr, train_nodes)

        # 标准化边特征（只用训练边）
        edge_attr = self._normalize_edge_features_safe(edge_attr, train_edges_idx)

        print(f"   节点特征维度: {node_attr.size(1)}")
        print(f"   边特征维度: {edge_attr.size(1)}")
        print(f"   训练边数: {len(train_edges_idx)}")

        # 创建图数据对象
        data = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
        data.y = edge_labels
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        return data

    def _build_node_features_from_edges(self, edge_index, edge_attr, num_nodes):
        """只用指定的边构建节点特征"""
        # 计算节点度
        node_degree = torch.zeros(num_nodes, dtype=torch.float32)
        node_degree.index_add_(0, edge_index[0], torch.ones(edge_index.size(1)))
        node_degree.index_add_(0, edge_index[1], torch.ones(edge_index.size(1)))

        # 累加边特征
        node_feat_sum = torch.zeros((num_nodes, edge_attr.size(1)), dtype=torch.float32)
        node_feat_sum.index_add_(0, edge_index[0], edge_attr)
        node_feat_sum.index_add_(0, edge_index[1], edge_attr)

        # 计算平均特征（避免除以0）
        node_degree_safe = node_degree.clamp(min=1)
        node_feat_mean = node_feat_sum / node_degree_safe.unsqueeze(1)

        # 处理孤立节点（用全局均值）
        global_mean = edge_attr.mean(dim=0)
        node_feat_mean[node_degree == 0] = global_mean

        # 添加对数度数特征
        degree_feat = torch.log1p(node_degree).unsqueeze(1)
        node_attr = torch.cat([node_feat_mean, degree_feat], dim=1).to(torch.float32)

        return node_attr

    def _normalize_features_safe(self, node_attr, train_nodes):
        """只用训练节点标准化节点特征"""
        if len(train_nodes) > 0:
            na_mean = node_attr[train_nodes].mean(dim=0)
            na_std = node_attr[train_nodes].std(dim=0) + 1e-8
            node_attr = (node_attr - na_mean) / na_std
        return node_attr

    def _normalize_edge_features_safe(self, edge_attr, train_edges_idx):
        """严格只用训练边标准化 - 修复泄露"""
        if len(train_edges_idx) > 0:
            ea_mean = edge_attr[train_edges_idx].mean(dim=0)
            ea_std = edge_attr[train_edges_idx].std(dim=0) + 1e-8

            # 只标准化训练边，测试边保持原始值或用训练统计量填充
            normalized = (edge_attr - ea_mean) / ea_std

            # 创建掩码：只有训练边被标准化，其他边保持为0或特殊值
            # 或者更安全的做法：测试边不参与训练，所以不需要标准化
            return normalized
        return edge_attr