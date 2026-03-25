import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, SAGEConv, GATConv
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = str(42)

# ================= 增强版MAF模块 =================
class MAF(torch.nn.Module):
    """增强版多尺度注意力融合模块"""

    def __init__(self, hidden_dim: int, edge_dim: int, num_heads: int, dropout: float = 0.3):
        super().__init__()
        #短程依赖捕获 - 使用图卷积网络（SAGEConv）
        self.short_gcn = SAGEConv(hidden_dim, hidden_dim)  #聚合直接邻居的信息（1跳邻居）

        #长程依赖捕获 - 使用Transformer卷积（能捕获全局信息）
        self.long_attn = TransformerConv(
            hidden_dim, hidden_dim, heads=num_heads,        #多头注意力，捕获远距离依赖
            concat=False, edge_dim=edge_dim,                #不拼接多头结果，使用边特征
            dropout=dropout                                 #Dropout比率
        )

        #邻居注意力机制 - 使用图注意力网络（GATConv）
        self.neighbor_attn = GATConv(
            hidden_dim, hidden_dim, heads=num_heads,        #多头注意力，关注重要的邻居
            concat=False, dropout=dropout                   #不拼接多头结果
        )

        #可学习的门控权重（控制三种特征的融合比例, 模型自己学）
        self.gate_short = torch.nn.Parameter(torch.ones(1))     #短程特征的权重（初始为1）
        self.gate_attn = torch.nn.Parameter(torch.ones(1))      #长程特征的权重（初始为1）
        self.gate_neighbor = torch.nn.Parameter(torch.ones(1))  #邻居注意力特征的权重（初始为1）

        # 特征融合层 - 把三种特征拼起来，压缩成最终表示
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 3, hidden_dim),        #384维 → 128维（压缩）
            torch.nn.LayerNorm(hidden_dim),                     #层归一化，稳定训练
            torch.nn.ReLU(),                                    #激活函数
            torch.nn.Dropout(dropout)                           #Dropout，防止过拟合
        )

        #层归一化（稳定特征分布）
        self.norm = torch.nn.LayerNorm(hidden_dim)              #对每一层的输出做归一化

        #Dropout层（防止过拟合）
        self.dropout = torch.nn.Dropout(dropout)                #随机丢弃神经元

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        #局部结构    [776,128]
        x_short = F.relu(self.short_gcn(x, edge_index))     #1.聚合邻居信息 2.激活函数（负数变0，正数不变）
        x_short = self.dropout(x_short)                     #3.防止过拟合

        #长程依赖    [776,128]
        x_attn = F.relu(self.long_attn(x, edge_index, edge_attr))   #1.Transformer捕获长程依赖（全局视野） 2.激活函数（负数变0，正数不变）
        x_attn = self.dropout(x_attn)                               #3.防止过拟合

        #邻居注意力   [776,128]
        x_neighbor = F.relu(self.neighbor_attn(x, edge_index))  #1.注意力图卷积（关注重要邻居）   2.激活函数
        x_neighbor = self.dropout(x_neighbor)                   #3.防止过拟合

        #门控融合
        gate_short = torch.sigmoid(self.gate_short)         #短程特征的开度（0~1）
        gate_attn = torch.sigmoid(self.gate_attn)           #长程特征的开度
        gate_neighbor = torch.sigmoid(self.gate_neighbor)   #邻居注意力的开度

        #融合后的节点特征
        x_fused = self.fusion(torch.cat([
            x_short * gate_short,       #加权后的短程特征[776, 128]
            x_attn * gate_attn,         #加权后的长程特征[776, 128]
            x_neighbor * gate_neighbor  #加权后的邻居注意力特征[776, 128]
        ], dim=1))                      #沿着特征维度拼接 → [776, 384) → 融合 → [776, 128]

        return self.norm(x + x_fused)  #残差连接:原始特征+新特征   层归一化：稳定数值，加速训练