import os
import numpy as np
from torch_geometric.data import Data #从 PyG 导入图数据结构 Data（存节点、边、特征）
import warnings
from config import Config

warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = str(42)

# ================= 边批量加载器 =================
class EdgeBatchLoader:
    """边级别的小批量加载器"""

    def __init__(self, data: Data, config: Config, shuffle: bool = True):
        self.data = data                            #存储原始图数据（节点、边、特征、标签等）
        self.batch_size = config.batch_size         #每批处理边数（如10000条）
        self.shuffle = shuffle                      #是否打乱边的顺序（训练打乱，验证不打乱）
        self.num_edges = data.edge_index.size(1)    #边的总数（总共有多少条通信记录）
        self.indices = np.arange(self.num_edges)    #边索引数组 [0,1,2,...,总边数-1]

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)         #随机打乱索引数组的顺序

        for start in range(0, self.num_edges, self.batch_size): #start = 0, 10000, 20000, 30000, ...  (每次跳 batch_size 步)
            end = min(start + self.batch_size, self.num_edges)  #结束位置 = 开始位置 + 批次大小
            batch_idx = self.indices[start:end]     #取出索引数组中[start,end)这批边在原始数据中的索引

            #创建一个新的图数据对象，只包含当前批次的数据
            batch_data = Data(
                x=self.data.x,                                  #节点特征（所有节点，不变）
                edge_index=self.data.edge_index[:, batch_idx],  #只取当前批次的边
                edge_attr=self.data.edge_attr[batch_idx],       #只取当前批次的边特征
                y=self.data.y[batch_idx]                        #只取当前批次的标签
            )

            if hasattr(self.data, 'train_mask'):                            #判断一个对象是否有某个属性
                batch_data.train_mask = self.data.train_mask[batch_idx]     #当前批次中每条边是否用于训练的布尔值
            if hasattr(self.data, 'val_mask'):
                batch_data.val_mask = self.data.val_mask[batch_idx]
            if hasattr(self.data, 'test_mask'):
                batch_data.test_mask = self.data.test_mask[batch_idx]

            yield batch_data        #返回一个批次数据，并暂停，等下次调用时继续

    def __len__(self):
        return (self.num_edges + self.batch_size - 1) // self.batch_size    #批次数量