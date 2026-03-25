import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data #从 PyG 导入图数据结构 Data（存节点、边、特征）
from sklearn.metrics import f1_score
import warnings
import copy
from typing import Dict

from config import Config
from edge_batch_loader import EdgeBatchLoader
from adaptive_threshold_optimizer import AdaptiveThresholdOptimizer

warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = str(42)

# ================= 训练器 =================
class Trainer:
    """优化版训练器"""

    def __init__(self, model: torch.nn.Module, config: Config,
                 attack_names: Dict[int, str], device: torch.device):
        self.model = model
        self.config = config
        self.attack_names = attack_names
        self.device = device

        self.best_val_f1 = 0        #到目前为止最好的验证集F1分数
        self.best_state = None      #最好模型时的参数（权重）
        self.best_thresholds = None #最好模型时的分类阈值
        self.patience_cnt = 0       #连续多少轮没有提升了（早停计数器）

        self.train_losses = []  #每个epoch的训练损失
        self.val_f1s = []       #每个epoch的验证集F1分数
        self.test_f1s = []      #每个epoch的测试集F1分数

    def train_epoch(self, loader, criterion, optimizer):
        """训练一个epoch"""
        self.model.train()  #设置模型为训练模式
        total_loss = 0      #累计所有批次的损失总和
        num_batches = 0     #累计处理的批次数量

        for batch_data in loader:                       #遍历数据加载器，每次取一个批次
            batch_data = batch_data.to(self.device)     #将批次数据移动到指定设备（CPU或GPU）
            optimizer.zero_grad()                       #清空上一批次的梯度，防止累积

            #每条边属于各个攻击类型的原始分数 [10000,10]
            out = self.model(batch_data.x, batch_data.edge_index, batch_data.edge_attr)  #前向传播：模型预测

            if hasattr(batch_data, 'train_mask') and batch_data.train_mask.sum() > 0:  #如果有训练掩码，且有训练样本
                loss = criterion(out[batch_data.train_mask], batch_data.y[batch_data.train_mask])  # 只对训练边计算损失 用损失函数计算预测和真实标签之间的差距
                loss.backward()     #反向传播，计算梯度（告诉模型该怎么调整）

                #梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  #限制梯度的大小，防止梯度爆炸

                optimizer.step()  #更新模型参数，根据计算好的梯度调整模型的权重和偏置

                total_loss += loss.item()   #累加当前批次的损失值
                num_batches += 1            #批次计数+1

            del batch_data  #       删除当前批次，释放内存

        return total_loss / max(num_batches, 1)     #返回一个 epoch 的平均损失

    @torch.no_grad()
    def evaluate(self, loader, mask_type='val'):
        """评估 - 原版（Transductive）"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        for batch_data in loader:
            batch_data = batch_data.to(self.device)

            # 原版：使用全量边（不区分训练/测试）
            out = self.model(batch_data.x, batch_data.edge_index, batch_data.edge_attr)
            probs = F.softmax(out, dim=1)
            preds = out.argmax(dim=1)

            mask = getattr(batch_data, f'{mask_type}_mask', None)
            if mask is not None and mask.sum() > 0:
                all_preds.append(preds[mask].cpu())
                all_labels.append(batch_data.y[mask].cpu())
                all_probs.append(probs[mask].cpu())

            del batch_data

        if all_preds:
            return (torch.cat(all_preds), torch.cat(all_labels), torch.cat(all_probs))
        return None, None, None

    def train(self, data: Data, criterion, optimizer, scheduler):
        """主训练循环"""
        print(f"\n[3/4] 开始训练...")

        train_loader = EdgeBatchLoader(data, self.config, shuffle=True)     #训练数据加载器（打乱顺序）
        val_loader = EdgeBatchLoader(data, self.config, shuffle=False)      #验证数据加载器（不打乱）

        for epoch in range(1, self.config.epochs + 1):
            #训练
            loss = self.train_epoch(train_loader, criterion, optimizer)
            self.train_losses.append(loss)      #每个epoch的训练损失

            #评估
            if epoch % 5 == 0 or epoch == 1:
                val_preds, val_labels, val_probs = self.evaluate(val_loader, 'val')     #评估验证集，从验证集中获取：预测结果、真实标签、预测概率
                test_preds, test_labels, test_probs = self.evaluate(val_loader, 'test') #评估测试集，从测试集中获取：预测结果、真实标签、预测概率

                if val_preds is not None:           #如果验证集有预测结果
                    val_f1 = f1_score(              #计算验证集F1分数
                        val_labels.numpy(),         #真实标签（转为 NumPy 数组）
                        val_preds.numpy(),          #预测标签（转为 NumPy 数组）
                        average='macro',            #宏平均：每个类别平等对待，适合不平衡数据集
                        zero_division=0             #如果某个类别没有预测到，F1=0，避免除以0报错
                    )
                else:                               #如果验证集没有预测结果
                    val_f1 = 0                      #F1分数设为0

                if test_preds is not None:          #如果测试集有预测结果（不为空）
                    test_f1 = f1_score(             #计算测试集F1分数
                        test_labels.numpy(),        #真实标签
                        test_preds.numpy(),         #预测标签
                        average='macro',            #宏平均：每个类别平等对待
                        zero_division=0             #避免除以0报错
                    )
                else:                               #如果测试集没有预测结果
                    test_f1 = 0                     #F1分数设为0

                self.val_f1s.append(val_f1)     #把当前轮次的验证集 F1 分数添加到历史列表中
                self.test_f1s.append(test_f1)   #把当前轮次的测试集 F1 分数添加到历史列表中

                # 学习率调整（基于验证集F1）
                if scheduler is not None:       #如果调度器存在
                    if isinstance(scheduler,torch.optim.lr_scheduler.ReduceLROnPlateau):  #如果是 ReduceLROnPlateau 类型的调度器
                        scheduler.step(val_f1)  #需要传入验证集 F1 分数（它根据指标是否提升来决定是否降学习率）
                    else:                       #如果是其他类型的调度器
                        scheduler.step()        #其他调度器直接步进,不需要传参数

                # 早停检查
                if val_f1 > self.best_val_f1:       #如果当前验证集 F1 比历史最高还要高
                    self.best_val_f1 = val_f1       #更新历史最高 F1 分数
                    self.best_state = copy.deepcopy(self.model.state_dict())    #深拷贝当前模型参数（保存最佳模型）

                    if self.config.use_adaptive_threshold and val_probs is not None:        #如果开启了自适应阈值，且验证集概率不为空
                        threshold_optimizer = AdaptiveThresholdOptimizer(self.config.threshold_strategy)    #创建自适应阈值优化器

                        #计算最佳分类阈值
                        self.best_thresholds = threshold_optimizer.fit(
                            val_labels.numpy(),     #验证集真实标签
                            val_probs.numpy(),      #验证集预测概率
                            self.attack_names       #攻击类型名称映射
                        )

                    self.patience_cnt = 0
                else:
                    self.patience_cnt += 1      #如果连续没提升的轮数（早停计数器）

                if epoch % 20 == 0:
                    print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val F1: {val_f1:.4f} | "  #轮次，损失，验证集F1分数
                          f"Test F1: {test_f1:.4f} | Best Val: {self.best_val_f1:.4f}")     #测试集F1分数，历史最好的验证集F1分数

                if self.patience_cnt >= self.config.patience:       #如果连续没提升的轮数 ≥ 设定的耐心值
                    print(f"⏱️ 早停于 Epoch {epoch}")
                    break
            else:
                #非评估epoch，但学习率调度器可能需要step
                if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()        #非评估轮次没有 val_f1

        #加载最佳模型
        if self.best_state:
            self.model.load_state_dict(self.best_state)     #把训练过程中保存的最佳模型参数加载到当前模型中
            print(f"\n✅ 加载最佳验证集模型 (Val F1: {self.best_val_f1:.4f})")

        return self.model, self.best_thresholds