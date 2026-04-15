import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from datetime import datetime
import time         #计时
from metrics_calculator import MetricsCalculator
import warnings
import json
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
from data_processor import DataProcessor
from config import Config as MainConfig


# ================= 配置（完全对齐主函数） =================
class LSTMBaselineConfig:
    """LSTM基线配置 - 与主函数GraphTransformer完全对齐"""
    # 模型架构（对齐）
    hidden_dim: int = 128   #对齐 GraphTransformer.hidden_channels
    num_layers: int = 2     #对齐 GraphTransformer.num_layers
    dropout: float = 0.3    #对齐 GraphTransformer.dropout

    # 训练参数（对齐）
    epochs: int = 300   #对齐
    patience: int = 50  #对齐
    learning_rate: float = 0.0003   #对齐 3e-4
    batch_size: int = 10000         #对齐
    weight_decay: float = 5e-4      #对齐

    # LSTM特有（但保持公平）
    bidirectional: bool = True  # 双向，使参数量接近Transformer
    num_lstm_layers: int = 2    # LSTM层数，对齐

    # 损失函数（对齐）
    focal_gamma: float = 2.0        #对齐 base_focal_gamma
    label_smoothing: float = 0.1    #对齐

    #路径
    data_path: str = "D:\\01Thesis\\04Git_project\\data\\train_test_network.csv"
    output_dir: str = "D:\\01Thesis\\04Git_project\\results"

    def __post_init__(self):
        """创建实验专用的输出目录"""
        os.makedirs(self.output_dir, exist_ok=True)                 #创建根目录
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")   #时间戳
        self.run_dir = os.path.join(self.output_dir, f"lstm_run_{self.timestamp}") #本次实验目录
        os.makedirs(self.run_dir, exist_ok=True)                    #创建实验目录

    def save(self):
        """保存配置"""
        config_path = os.path.join(self.run_dir, 'config.yaml') #将配置保存为 YAML 文件，便于复现实验
        with open(config_path, 'w') as f:
            yaml.dump(self.__dict__, f)

# ================= LSTM模型（对齐Transformer分类头） =================
class LSTMBaseline(nn.Module):
    """
    LSTM基线模型 - 严格对齐GraphTransformer

    对齐策略：
    1. 隐藏层维度：128（对齐）
    2. 层数：2（对齐）
    3. Dropout：0.3（对齐）
    4. 分类器结构：对齐Transformer的classifier
    5. 使用Focal Loss（对齐）
    """

    def __init__(self, input_dim: int, num_classes: int, config: LSTMBaselineConfig):
        super().__init__()

        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.bidirectional = config.bidirectional
        self.num_directions = 2 if config.bidirectional else 1

        # ===== 输入编码器（对齐Transformer的node_encoder结构）=====
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),  #11 → 128
            nn.BatchNorm1d(config.hidden_dim),        #归一化
            nn.ReLU(),                                #激活函数
            nn.Dropout(config.dropout)                #Dropout防止过拟合
        )

        # ===== LSTM层 =====
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,   #输入维度 128
            hidden_size=config.hidden_dim,  #隐藏层维度 128
            num_layers=2,                   #2层LSTM
            batch_first=True,               #批次维度在第一维
            bidirectional=True              #双向LSTM（参数量翻倍）
        )

        # ===== 分类器（完全对齐Transformer的分类器结构）=====
        # Transformer分类器: Linear(hidden*3, hidden*2) -> BN -> ReLU -> Dropout
        #                    -> Linear(hidden*2, hidden) -> BN -> ReLU -> Dropout
        #                    -> Linear(hidden, num_classes)
        #
        # LSTM使用 hidden * num_directions 对齐 Transformer 的 hidden * 3

        lstm_out_dim = config.hidden_dim * self.num_directions

        self.classifier = nn.Sequential(
            # 第1层：256 → 256
            nn.Linear(lstm_out_dim, config.hidden_dim * 2),
            nn.BatchNorm1d(config.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),

            # 第2层：256 → 128
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout // 2),
        )
        # 输出层
        self.output_proj = nn.Linear(config.hidden_dim, num_classes)  # 128 → 10

        #初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重（对齐Transformer）"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        参数:
            x: 边特征 [batch_size, input_dim]
        返回:
            logits: [batch_size, num_classes]
        """
        #1. 输入编码
        x = self.input_encoder(x)  # [batch, 128]

        #2. LSTM需要序列维度，将每个样本视为长度为1的序列
        #    这样LSTM退化为带门控的全连接层（公平对比）
        x = x.unsqueeze(1)  # [batch, 1, 128]

        #3. LSTM前向
        lstm_out, (h_n, c_n) = self.lstm(x)  #[batch, 1, 256]（双向=256）

        #4. 取最后时间步的输出
        out = lstm_out[:, -1, :]    #[batch, 128 * 2]

        #5. 分类器
        out = self.classifier(out)  #[batch, 128]

        #6. 输出层
        logits = self.output_proj(out)  #[batch, 10]

        return logits

# ================= Focal Loss（对齐主函数） =================
class FocalLoss(nn.Module):
    """Focal Loss - 与主函数完全对齐"""

    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        # 标签平滑
        n_classes = input.size(1)
        log_probs = F.log_softmax(input, dim=1)

        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs,
                                             self.label_smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, target.unsqueeze(1),
                                    1.0 - self.label_smoothing)

        # 交叉熵
        ce = -(smooth_targets * log_probs).sum(dim=1)

        # Focal权重
        pt = torch.exp(-ce)
        focal_weight = (1 - pt) ** self.gamma

        # 类别权重
        if self.weight is not None:
            sample_weight = self.weight[target]
            loss = (focal_weight * ce * sample_weight).mean()
        else:
            loss = (focal_weight * ce).mean()

        return loss

# ================= 训练器（对齐主函数Trainer） =================
class LSTMTrainer:
    """LSTM训练器 - 与主函数Trainer完全对齐"""

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.best_val_f1 = 0
        self.best_state = None
        self.patience_cnt = 0

        # 记录训练历史
        self.train_losses = []
        self.val_f1s = []
        self.test_f1s = []

    def train_epoch(self, loader, criterion, optimizer):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()

            # 梯度裁剪（对齐）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self, loader):
        """评估模型"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            logits = self.model(X_batch)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        if all_preds:
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            return np.array(all_preds), np.array(all_labels), np.array(all_probs), f1
        return None, None, None, 0.0

    def train(self, train_loader, val_loader, test_loader, criterion, optimizer, scheduler):
        """主训练循环"""
        print(f"\n🚀 开始训练LSTM基线...")
        print(f"   参数: epochs={self.config.epochs}, lr={self.config.learning_rate}")

        for epoch in range(1, self.config.epochs + 1):
            # 训练
            loss = self.train_epoch(train_loader, criterion, optimizer)
            self.train_losses.append(loss)

            # 每5个epoch评估一次（对齐主函数）
            if epoch % 5 == 0 or epoch == 1:
                # 验证集评估
                val_preds, val_labels, val_probs, val_f1 = self.evaluate(val_loader)
                # 测试集评估
                test_preds, test_labels, test_probs, test_f1 = self.evaluate(test_loader)

                self.val_f1s.append(val_f1)
                self.test_f1s.append(test_f1)

                # 学习率调度（对齐主函数）
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_f1)
                    else:
                        scheduler.step()

                # 早停检查
                if val_f1 > self.best_val_f1:
                    self.best_val_f1 = val_f1
                    self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    self.patience_cnt = 0
                else:
                    self.patience_cnt += 1

                # 打印进度
                if epoch % 20 == 0:
                    print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
                          f"Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f} | "
                          f"Best Val: {self.best_val_f1:.4f}")

                # 早停
                if self.patience_cnt >= self.config.patience:
                    print(f"⏱️ 早停于 Epoch {epoch}")
                    break

        # 加载最佳模型
        if self.best_state:
            self.model.load_state_dict(self.best_state)
            print(f"\n✅ 加载最佳验证集模型 (Val F1: {self.best_val_f1:.4f})")

        return self.model

def save_results(metrics, model, config, attack_names, train_history, test_labels, test_preds, test_probs):
    """保存所有结果"""
    save_dir = config.run_dir

    # 1. 保存指标
    results = {
        'timestamp': config.timestamp,
        'model': 'LSTMBaseline',
        'configuration': {
            'hidden_dim': config.hidden_dim,
            'num_layers': config.num_layers,
            'dropout': config.dropout,
            'bidirectional': config.bidirectional,
            'epochs': config.epochs,
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'weight_decay': config.weight_decay,
            'focal_gamma': config.focal_gamma,
            'label_smoothing': config.label_smoothing,
        },
        'metrics': metrics,
        'training_history': train_history
    }

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   ✓ results.json")

    # 2. 保存模型
    model_path = os.path.join(save_dir, 'model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'attack_names': attack_names,
        'metrics': metrics
    }, model_path)
    print(f"   ✓ model.pth")

    # 3. 保存混淆矩阵图
    plot_confusion_matrix(test_labels, test_preds, attack_names, save_dir)
    print(f"   ✓ confusion_matrix.png")

    # 4. 保存训练曲线
    plot_training_curves(train_history, save_dir)
    print(f"   ✓ training_curves.png")

def plot_confusion_matrix(y_true, y_pred, attack_names, save_dir):
    """绘制混淆矩阵"""
    plt.figure(figsize=(14, 6))

    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    target_names = [attack_names.get(i, f'Class_{i}') for i in unique_labels]

    # 原始计数
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title("LSTM Baseline - Confusion Matrix (Counts)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45)

    # 归一化
    plt.subplot(1, 2, 2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title("LSTM Baseline - Confusion Matrix (Normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_training_curves(train_history, save_dir):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 4))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_history.get('train_loss', []), label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # F1 曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_history.get('val_f1', []), label='Val F1')
    plt.plot(train_history.get('test_f1', []), label='Test F1')
    plt.xlabel('Epoch')
    plt.ylabel('Macro-F1')
    plt.title('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

# ================= 使用示例 =================
if __name__ == "__main__":
    #创建配置
    lstm_config = LSTMBaselineConfig()

    #复用主函数的数据处理器
    main_config = MainConfig()
    main_config.data_path = lstm_config.data_path
    processor = DataProcessor(main_config)

    #加载数据
    df = processor.load_data()
    df, train_idx, val_idx, test_idx, attack_names, feature_cols = processor.preprocess(df)

    # 提取特征和标签（边级别）
    X = df[feature_cols].values     #从DataFrame中提取特征数据，转换为NumPy数组
    y = df['attack_type'].values    #从DataFrame中提取标签数据，转换为NumPy数组

    X_train = X[train_idx]  #使用预划分的索引，从总数据中取出训练集的特征
    X_val = X[val_idx]      #从总数据中取出验证集的特征
    X_test = X[test_idx]    #~测试集的特征
    y_train = y[train_idx]  #~训练集的标签
    y_val = y[val_idx]      #~验证集的标签
    y_test = y[test_idx]    #~测试集的标签

    #将提取的特征和标签转换为张量
    X_train_t = torch.FloatTensor(X_train)
    X_val_t = torch.FloatTensor(X_val)
    X_test_t = torch.FloatTensor(X_test)
    y_train_t = torch.LongTensor(y_train)
    y_val_t = torch.LongTensor(y_val)
    y_test_t = torch.LongTensor(y_test)

    #创建DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t) #将训练集的特征和标签打包成数据集
    val_dataset = TensorDataset(X_val_t, y_val_t)       #验证集
    test_dataset = TensorDataset(X_test_t, y_test_t)    #测试集

    train_loader = DataLoader(train_dataset, batch_size=lstm_config.batch_size, shuffle=True)#创建训练数据加载器：批量大小=配置值（如10000），训练时打乱顺序
    val_loader = DataLoader(val_dataset, batch_size=lstm_config.batch_size, shuffle=False)   #创建验证数据加载器：批量大小=配置值，不打乱顺序
    test_loader = DataLoader(test_dataset, batch_size=lstm_config.batch_size, shuffle=False) #创建测试数据加载器：批量大小=配置值，不打乱顺序

    #设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    #创建模型
    model = LSTMBaseline(
        input_dim=X_train.shape[1],     #输入特征维度（如 16）
        num_classes=len(attack_names),  #输出类别数（如 10 种攻击类型）
        config=lstm_config              #LSTM 配置参数（隐藏层维度、层数、dropout等）
    ).to(device)                        #将模型移动到CPU

    print(f"\n🤖 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    train_start_time = time.time()      #开始的时间

    #损失函数和优化器
    criterion = FocalLoss(              #创建 Focal Loss 损失函数（处理类别不平衡）
        gamma=lstm_config.focal_gamma,  #聚焦参数（4.0），控制对难样本的关注程度
        label_smoothing=lstm_config.label_smoothing #标签平滑（0.1），防止模型过自信
    )
    #创建 AdamW 优化器（更新模型参数）
    optimizer = torch.optim.AdamW(
        model.parameters(),                     #要优化的模型参数
        lr=lstm_config.learning_rate,           #学习率（0.0005），控制参数更新步长
        weight_decay=lstm_config.weight_decay   #权重衰减（1e-4），L2正则化防止过拟合
    )
    #创建学习率调度器（动态调整学习率）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,      #要调整学习率的优化器
        mode='max',     #监控指标越大越好（这里是验证集 F1）
        factor=0.5,     #学习率衰减因子（每次减半）
        patience=10,    #10个epoch没提升就降低学习率
        verbose=True    #打印学习率调整信息
    )

    #训练
    trainer = LSTMTrainer(model, lstm_config, device)       #创建 LSTM 训练器对象，负责管理整个训练流程
    model = trainer.train(train_loader, val_loader, test_loader,
                          criterion, optimizer, scheduler)  #开始训练模型，返回训练好的最佳模型

    #最终评估
    test_preds, test_labels, test_probs, test_f1 = trainer.evaluate(test_loader)
    print(f"\n🏆 最终测试集 Macro-F1: {test_f1:.4f}")

    # 评估时使用
    metrics = MetricsCalculator.calculate_all(
        y_true=test_labels,      #真实标签
        y_pred=test_preds,       #预测标签
        y_prob=test_probs,       #预测概率
        class_names=attack_names #类别名称映射
    )

    #获取指标
    macro_f1 = metrics['macro_f1']
    accuracy = metrics['accuracy']
    macro_fpr = metrics['macro_fpr']
    macro_fnr = metrics['macro_fnr']
    weighted_fpr = metrics['weighted_fpr']
    weighted_fnr = metrics['weighted_fnr']
    auc_roc = metrics.get('mean_auc_roc', 0)
    auc_pr = metrics.get('mean_auc_pr', 0)

    #打印
    print(f"🏆 最终 Macro-F1: {macro_f1:.4f}")
    print(f"📊 Accuracy: {accuracy:.4f}")
    print(f"📈 Mean AUC-ROC: {auc_roc:.4f}")
    print(f"📉 Mean AUC-PR: {auc_pr:.4f}")
    print(f"❌ 宏平均误报率 (Macro-FPR): {macro_fpr:.4f}")
    print(f"⚠️ 宏平均漏报率 (Macro-FNR): {macro_fnr:.4f}")
    print(f"⚖️ 加权误报率: {weighted_fpr:.4f}")
    print(f"⚖️ 加权漏报率: {weighted_fnr:.4f}")

    train_history = {
        'train_loss': trainer.train_losses,
        'val_f1': trainer.val_f1s,
        'test_f1': trainer.test_f1s
    }

    # 保存结果
    save_results(metrics, model, lstm_config, attack_names,
                 train_history, test_labels, test_preds, test_probs)

    print(f"\n💾 所有结果已保存至: {lstm_config.run_dir}")

    train_time = time.time() - train_start_time
    print(f"\n⏱️ 训练耗时: {train_time / 60:.2f} 分钟)")