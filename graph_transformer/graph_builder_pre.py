import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data #从 PyG 导入图数据结构 Data（存节点、边、特征）
import warnings
from typing import  List

warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = str(42)

# ================= 修复版图构建器 =================
class GraphBuilder:
    """图构建器 - 使用所有边构建节点特征"""

    def __init__(self):
        self.ip_to_id = {}

    def build(self, df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray,
              test_idx: np.ndarray, feature_cols: List[str]) -> Data:
        """构建图数据"""
        print("\n[2/4] 构建图结构...")

        # 创建节点映射
        unique_ips = pd.unique(pd.concat([df['src_ip'], df['dst_ip']]))     #包含所有出现过的IP地址的数组
        self.ip_to_id = {ip: i for i, ip in enumerate(unique_ips)}      #{IP: 索引号} 的映射关系
        num_nodes = len(unique_ips)     #获取节点总数
        print(f"   节点数: {num_nodes}")

        #将IP地址列中的所有IP转换为对应的数字节点
        src_ids = df['src_ip'].map(self.ip_to_id).values.astype(np.int64)
        dst_ids = df['dst_ip'].map(self.ip_to_id).values.astype(np.int64)

        #将网络流量数据转换为图神经网络所需的图结构数据
        edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long) #将源IP和目标IP的节点ID组合成边索引
        num_edges = edge_index.size(1)  # 统计总共有多少条网络连接（边）
        print(f"   边数: {num_edges}")
        edge_attr = torch.tensor(df[feature_cols].values.astype(np.float32), dtype=torch.float) #提取每条流量的攻击类型标签,[211043, 41]
        edge_labels = torch.tensor(df['attack_type'].values, dtype=torch.long)  #提取每条流量的攻击类型标签

        # 创建masks
        train_mask = torch.zeros(num_edges, dtype=torch.bool)
        val_mask = torch.zeros(num_edges, dtype=torch.bool)
        test_mask = torch.zeros(num_edges, dtype=torch.bool)

        #训练集、验证集、测试集的样本位置标记为 True
        train_mask[torch.tensor(train_idx, dtype=torch.long)] = True
        val_mask[torch.tensor(val_idx, dtype=torch.long)] = True
        test_mask[torch.tensor(test_idx, dtype=torch.long)] = True

        # === 修复点：使用所有边构建节点特征 ===
        node_attr = self._build_node_features(edge_index, edge_attr, num_nodes)

        # 标准化（仅基于训练节点）
        node_attr, edge_attr = self._normalize_features(node_attr, edge_attr, edge_index, train_mask)
        print(f"   节点特征维度: {node_attr.size(1)}")
        print(f"   边特征维度: {edge_attr.size(1)}")

        #创建图数据对象data，将节点特征、边索引和边特征封装在一起
        data = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
        data.y = edge_labels
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        return data

    def _build_node_features(self, edge_index, edge_attr, num_nodes):
        """修复版：使用所有边构建节点特征"""
        #计算图中每个节点的度（Degree）,[1,776]
        node_degree = torch.zeros(num_nodes, dtype=torch.float32)       #度向量统计每个节点有多少条边
        node_degree.index_add_(0, edge_index[0], torch.ones(edge_index.size(1)))      #出度
        node_degree.index_add_(0, edge_index[1], torch.ones(edge_index.size(1)))      #入度

        #节点特征累加器       [节点数776，特征数41]
        node_feat_sum = torch.zeros((num_nodes, edge_attr.size(1)), dtype=torch.float32)
        node_feat_sum.index_add_(0, edge_index[0], edge_attr)   #将每条边的特征累加到其源节点和目标节点的特征总和中
        node_feat_sum.index_add_(0, edge_index[1], edge_attr)

        #计算每个节点的平均特征，并避免除以0     [776,41]
        node_degree_safe = node_degree.clamp(min=1)     #将小于1的值设为1，大于1的不变
        node_feat_mean = node_feat_sum / node_degree_safe.unsqueeze(1)  #逐元素相除，每个节点的特征总和除以该节点的度数

        #处理孤立节点（使用全局均值）
        global_mean = edge_attr.mean(dim=0)     #对所有边的特征求平均，得到每个特征的全局平均值(列平均值) [1,41]
        node_feat_mean[node_degree == 0] = global_mean      #将度数为0的节点的特征设置为全局平均值

        #对节点度数取对数变换  [776,1]
        degree_feat = torch.log1p(node_degree).unsqueeze(1)     #对节点度数进行对数变换，使数值更平滑

        #将 节点平均特征 和 节点度数特征 在列维度上拼接，合并成一个完整的 节点特征矩阵[776,42]
        node_attr = torch.cat([node_feat_mean, degree_feat], dim=1).to(torch.float32)

        return node_attr

    def _normalize_features(self, node_attr, edge_attr, edge_index, train_mask):
        """修复版：仅基于训练节点标准化"""
        # 获取参与训练过的节点train_nodes
        train_edges = torch.where(train_mask)[0]
        if len(train_edges) > 0:
            train_nodes = torch.unique(torch.cat([
                edge_index[0, train_edges],         #选取第一行（源节点）中，被 train_edges 标记的列
                edge_index[1, train_edges]          #选取第二行（目标节点）中，被 train_edges 标记的列
            ]))

            if len(train_nodes) > 0:
                # 节点特征标准化
                na_mean = node_attr[train_nodes].mean(dim=0)     #计算所有训练节点在每一维特征上的平均值
                na_std = node_attr[train_nodes].std(dim=0) + 1e-8  #计算所有训练节点在每一维特征上的标准差，然后加上一个极小值（1e-8）防止除零错误
                node_attr = (node_attr - na_mean) / na_std            #特征标准化，将数据转换为均值为0、标准差为1的分布

                # 边特征标准化
                ea_mean = edge_attr[train_edges].mean(dim=0)
                ea_std = edge_attr[train_edges].std(dim=0) + 1e-8
                edge_attr = (edge_attr - ea_mean) / ea_std

        return node_attr, edge_attr