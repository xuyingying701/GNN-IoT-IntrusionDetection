import os
import torch
import warnings
from maf import MAF
warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = str(42)

# ================= 优化版图Transformer =================
class GraphTransformer(torch.nn.Module):
    """优化版图Transformer模型"""

    def __init__(self, in_ch: int, edge_ch: int, hidden: int, out_ch: int,
                 heads: int, layers: int, dropout: float = 0.3):
        super().__init__()
        #创建一个节点特征编码器(将原始的节点特征转换为更高维度的表示)
        self.node_encoder = torch.nn.Sequential(
            torch.nn.Linear(in_ch, hidden),     #1.线性变换层
            torch.nn.BatchNorm1d(hidden),       #2.批归一化层
            torch.nn.ReLU(),                    #3.激活函数层
            torch.nn.Dropout(dropout)           #4.Dropout层
        )

        #创建一个边特征编码器，将原始的边特征（通信特征）转换为更高维度的表示
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(edge_ch, hidden),   #1.线性变换层
            torch.nn.BatchNorm1d(hidden),       #2.批归一化层
            torch.nn.ReLU(),                    #3.激活函数层
            torch.nn.Dropout(dropout)           #4.Dropout层
        )

        #创建一个模块列表，用于存储多个MAF（Multi-head Attention Fusion多头注意力融合层）层
        self.layers = torch.nn.ModuleList()     #创建模块列表容器
        for _ in range(layers):                 #循环layers次
            self.layers.append(MAF(hidden, hidden, heads, dropout))

        #创建一个多层分类器，将节点特征映射到最终的类别输出
        self.classifier = torch.nn.Sequential(
            #第1层：维度降低
            torch.nn.Linear(hidden * 3, hidden * 2),  # 384 → 256
            torch.nn.BatchNorm1d(hidden * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),  # 0.3

            #第2层：维度再降低
            torch.nn.Linear(hidden * 2, hidden),  # 256 → 128
            torch.nn.BatchNorm1d(hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout // 2),  # 0.1

            #输出层
            torch.nn.Linear(hidden, out_ch)  # 128 → 9
        )

        #初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""      #初始化神经网络中所有线性层的参数（让模型从更好的起点开始学习）
        for module in self.modules():                       # 遍历模型的所有层
            if isinstance(module, torch.nn.Linear):             #如果是线性层
                torch.nn.init.xavier_uniform_(module.weight)    #1.用xavier方法初始化权重
                if module.bias is not None:                     #2.偏置初始化为0（刚开始没有偏见，公平）
                    torch.nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        x = self.node_encoder(x)                    #每个节点的特征（高维表示）  [776,128]
        edge_feat = self.edge_encoder(edge_attr)    #每条通信的特征（高维表示)   [211043,128]

        for layer in self.layers:                   #循环遍历每一层 MAF（多头注意力融合层）
            x = layer(x, edge_index, edge_feat)     #通过MAF层，融合了邻居设备后的设备特征   [776,128]

        # 边分类
        src, dst = edge_index                       #拆解边索引：src = 源节点索引，dst = 目标节点索引  [10000, 128],[10000, 128],[10000, 128]
        edge_out = torch.cat([x[src], x[dst], edge_feat], dim=1)  #x[src]：取出源设备的特征，按第1维（列方向）进行拼接  [10000, 384]

        return self.classifier(edge_out)            #通过分类器，输出每条边属于各类别的分数 [10000, 10]