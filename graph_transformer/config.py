import os
import warnings
import yaml
from datetime import datetime
from typing import Dict,Any
from dataclasses import dataclass, field

warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = str(42)

# ================= 配置数据类 =================
@dataclass
class Config:
    """模型配置类"""
    #基础配置
    random_seed: int = 42
    hidden_channels: int = 128  #隐藏层维度
    num_heads: int = 4      #多头注意力头数
    num_layers: int = 2     #图神经网络层数
    epochs: int = 300       #最大训练 300 轮
    patience: int = 50      #早停：50 轮不升就停
    learning_rate: float = 0.0003  # 学习率 3e-4
    dropout: float = 0.3    #防止过拟合
    batch_size: int = 10000     #批次大小
    weight_decay: float = 5e-4  #权重衰减，正则化

    #策略开关
    use_curriculum: bool = False  # 课程学习关闭
    use_adaptive_threshold: bool = True     #自适应阈值开启
    use_class_specific_gamma: bool = True
    use_label_smoothing: bool = True  #标签平滑开启
    label_smoothing: float = 0.1      #标签平滑系数

    #Loss参数
    base_focal_gamma: float = 2.0

    #类别特定配置
    class_specific_boost: Dict[str, float] = field(default_factory=lambda: {
        'injection': 1.5,   #损失放大1.5倍
        'password': 1.5,
        'mitm': 1.2,
    })          #给难分类攻击加大权重

    class_gamma: Dict[str, float] = field(default_factory=lambda: {
        'injection': 2.0,
        'password': 2.5,
        'mitm': 2.0,
    })#不同攻击不同 focal gamma，实现对不同难度攻击的差异化关注

    #配置自适应阈值的优化策略
    threshold_strategy: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'injection': {'target': 'f1'},      #对注入攻击，用F1分数找最佳阈值
        'password': {'target': 'f1'},
        'mitm': {'target': 'f1'},
        'default': {'target': 'f1'}     #其他攻击类型，默认也用F1分数
    })          #为不同攻击类型自动调整分类阈值，提升整体检测效果

    #数据路径
    data_path: str = "D:\\01Thesis\\04Git_project\\data\\train_test_network.csv"
    output_dir: str = "D:\\01Thesis\\04Git_project\\results"

    def __post_init__(self):
        """创建实验专用的输出目录"""
        os.makedirs(self.output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.output_dir, f"graph_transformer_run_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

    def save(self):
        """保存配置"""
        config_path = os.path.join(self.run_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.__dict__, f)