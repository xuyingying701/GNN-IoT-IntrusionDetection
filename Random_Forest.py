import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier #导入随机森林分类器，这是核心模型
from sklearn.metrics import (                       #导入评估指标
    classification_report, accuracy_score, f1_score,#生成完整的分类报告，准确率，F1分数（精确率和召回率的调和平均）
    precision_score, recall_score, confusion_matrix,#精确率（预测为正例中实际为正的比例），召回率（实际为正例中被正确预测的比例），混淆矩阵
    roc_auc_score, average_precision_score
)
from sklearn.preprocessing import LabelEncoder, label_binarize,StandardScaler     #将类别标签（如"normal","attack"）转换为数字（0,1）
from sklearn.model_selection import train_test_split#数据集划分函数
from metrics_calculator import MetricsCalculator
import matplotlib
matplotlib.use('Agg')       #Agg表示无图形界面后端（适合服务器运行）
import matplotlib.pyplot as plt
import seaborn as sns       #基于matplotlib的统计可视化库
import json         #保存结果指标
import joblib       #保存训练好的模型
import os
import sys
import warnings     #忽略警告信息
import time         #计时
import traceback    #错误堆栈跟踪
from datetime import datetime   #时间戳

warnings.filterwarnings("ignore")

# ==========================================
# 配置部分 (可根据数据集调整)
# ==========================================

#【路径配置】
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) #当前脚本所在目录
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)              #项目根目录（上一级）
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'train_test_network.csv')    #数据文件完整路径 项目根目录/data/train_test_network.csv
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results')           #结果保存目录 项目根目录/results
os.makedirs(OUTPUT_DIR, exist_ok=True)          #创建输出目录

#【实验配置】
RANDOM_STATE = 42           #随机种子，保证结果可复现
TEST_SIZE = 0.2             #测试集占全部数据的20%
VAL_SIZE = 0.2              #验证集占全部数据的20%

TARGET_COLUMN = 'type'          #标签列的名字叫 'type'
NORMAL_CLASS_NAME = 'normal'    #正常流量的类别名称是 'normal'

#【模型配置】
N_ESTIMATORS = 200      #森林中决策树的数量（要种多少棵树）
MAX_DEPTH = 30          #每棵树的最大深度（每个专家能思考多深），防止过拟合
CLASS_WEIGHT = 'balanced'#自动给"少数类"更高的权重，让模型更关注它们

# ==========================================
# 工具函数
# ==========================================

def save_results(results_dict, y_test, y_pred, class_names, timestamp):
    """
    保存所有结果到文件
    """
    run_dir = os.path.join(OUTPUT_DIR, f"rf_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)         #创建这个目录。如果目录已存在，不报错，直接继续；如果不存在，则创建

    print(f"\n💾 保存结果到: {run_dir}")

    #1.打开文件，准备写入JSON数据
    with open(os.path.join(run_dir, 'metrics.json'), 'w', encoding='utf-8') as f:   #打开文件，赋值给变量 f，退出自动关闭
        json.dump(results_dict, f, indent=2, ensure_ascii=False)    #将Python字典写入JSON文件
    print(f"   ✓ metrics.json")

    #2.保存分类报告
    #生成分类报告（包含每个类别的精确率、召回率、F1分数、样本数）
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)

    with open(os.path.join(run_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Random Forest Baseline - {timestamp}\n")      #写入标题行
        f.write("=" * 70 + "\n\n")      #写入分隔线（70个等号）
        f.write(report)     #写入分类报告内容
        # 添加核心指标
        f.write(f"\n\n{'=' * 50}\n")
        f.write("CORE METRICS\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Macro-F1: {results_dict['metrics']['macro_f1']:.4f}\n")
        f.write(f"Accuracy: {results_dict['metrics']['accuracy']:.4f}\n")
        f.write(f"Mean AUC-ROC: {results_dict['metrics']['auc_roc']:.4f}\n")
        f.write(f"Mean AUC-PR: {results_dict['metrics']['auc_pr']:.4f}\n")
        f.write(f"Macro-FPR: {results_dict['metrics']['macro_fpr']:.4f}\n")
        f.write(f"Macro-FNR: {results_dict['metrics']['macro_fnr']:.4f}\n")
        f.write(f"Weighted FPR: {results_dict['metrics']['weighted_fpr']:.4f}\n")
        f.write(f"Weighted FNR: {results_dict['metrics']['weighted_fnr']:.4f}\n")
    print(f"   ✓ classification_report.txt")    #分类报告成功保存

    #3.保存混淆矩阵数据
    cm = confusion_matrix(y_test, y_pred)   #计算混淆矩阵（真实标签 vs 预测标签）
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)#将混淆矩阵转换为DataFrame，并设置行索引和列索引为类别名称
    cm_df.to_csv(os.path.join(run_dir, 'confusion_matrix.csv'))     #将混淆矩阵保存为CSV文件

    #4.绘制混淆矩阵图
    # 创建一个12英寸宽、5英寸高的图形窗口
    plt.figure(figsize=(12, 5))

    # ========== 左图：原始计数混淆矩阵 ==========
    plt.subplot(1, 2, 1)  # 创建1行2列的子图，选中第1个（左图）
    sns.heatmap(    #绘制热力图
        cm,         #混淆矩阵数据（整数计数）
        annot=True, #在格子中显示数值
        fmt='d',    #数值格式为整数（d=decimal整数）
        cmap='Blues',  #颜色映射为蓝色系
        cbar=False,    #不显示颜色条
        xticklabels=class_names,    #X轴标签为类别名称
        yticklabels=class_names     #Y轴标签为类别名称
    )
    plt.title('Confusion Matrix (Counts)')  #左图标题：原始计数混淆矩阵
    plt.xlabel('Predicted')  #X轴标签：预测值
    plt.ylabel('True')       #Y轴标签：真实值

    # ========== 右图：归一化混淆矩阵 ==========
    plt.subplot(1, 2, 2)  #选中第2个（右图）
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # 每行归一化（每行和为1）
    sns.heatmap(
        cm_norm,        #归一化后的混淆矩阵（0~1的小数）
        annot=True,     #在格子中显示数值
        fmt='.2f',      #数值格式为保留2位小数
        cmap='Blues',   #颜色映射为蓝色系
        cbar=False,     #不显示颜色条
        xticklabels=class_names,  #X轴标签为类别名称
        yticklabels=class_names   #Y轴标签为类别名称
    )
    plt.title('Confusion Matrix (Normalized)')  #右图标题：归一化混淆矩阵
    plt.xlabel('Predicted')     #X轴标签：预测值
    plt.ylabel('True')          #Y轴标签：真实值

    # ========== 保存图片 ==========
    plt.tight_layout()  #自动调整子图间距，避免重叠
    plt.savefig(os.path.join(run_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')  #保存图片
    plt.close()
    print(f"   ✓ confusion_matrix.csv/png")

    return run_dir

# ==========================================
# 主程序
# ==========================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")#获取当前时间并格式化为"年月日_时分秒"字符串，用于生成唯一的时间戳标识

    print(f"\n{'='*70}")
    print("🌲 随机森林基线模型")
    print(f"{'='*70}")
    print(f"数据路径: {DATA_PATH}")
    print(f"随机种子: {RANDOM_STATE}")
    print(f"{'='*70}\n")

    try:
        # ---------- 1. 加载数据 ----------
        print(f"📂 [1/5] 加载数据...")
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"找不到文件: {DATA_PATH}")

        df = pd.read_csv(DATA_PATH)     #读出的数据
        print(f"   原始数据: {df.shape[0]} 行 × {df.shape[1]} 列")

        #检查目标列
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"目标列 '{TARGET_COLUMN}' 不存在!")

        #显示类别分布
        print(f"\n   类别分布:")
        target_counts = df[TARGET_COLUMN].value_counts()    #统计TARGET_COLUMN列中每个类别出现的次数
        for cls, count in target_counts.items():            #遍历每个类别及其对应的数量
            print(f"     {cls}: {count} ({count/len(df)*100:.2f}%)")    #打印：类别名称、数量、占比

        # ---------- 2. 预处理 ----------
        print(f"\n🔧 [2/5] 特征预处理...")

        #标签编码
        le_target = LabelEncoder()      #创建标签编码器对象
        y = le_target.fit_transform(df[TARGET_COLUMN])#将TARGET_COLUMN列中的类别文本（如'normal','attack'）转换为数字（0,1,2...），并返回编码后的数组
        class_names = le_target.classes_    #获取编码前的原始类别名称列表，如['normal', 'dos', 'scan']，索引0对应normal，索引1对应dos
        n_classes = len(class_names)        #统计类别总数（共有多少种攻击类型+正常）
        #print(n_classes)

        #找normal类索引
        normal_idx = list(class_names).index(NORMAL_CLASS_NAME)
        print(f"   🛡️ 正常类: '{NORMAL_CLASS_NAME}' (索引 {normal_idx})")    #索引下标从0开始

        feature_cols = [
            'duration', 'src_bytes', 'dst_bytes', 'src_pkts', 'dst_pkts',
            'missed_bytes', 'proto', 'service', 'ssl_version', 'http_method', 'conn_state',
            #'src_ip_bytes',     #源IP总字节数（反映源设备的流量总量）
            'dst_ip_bytes',     #目标IP总字节数（反映目标设备的流量总量）
            'http_status_code',  #HTTP状态码（区分正常/异常HTTP响应）
            'ssl_cipher',  # SSL加密套件（检测弱加密/恶意加密）
            'ssl_resumed',  # SSL会话恢复（检测异常会话）
            'dns_qtype'  # DNS查询类型（检测DNS隧道）
        ]
        #自动区分数值和类别
        NUMERIC_FEATURES = [c for c in feature_cols if df[c].dtype in ['int64', 'float64']]
        CATEGORICAL_FEATURES = [c for c in feature_cols if df[c].dtype == 'object']

        #特征处理
        df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0)       #将数值特征列中的所有空值填充为 0
        df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].fillna('Unknown')   #将类别特征列中的所有空值填充为 'Unknown'（字符串）

        selected_features = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES
                            if c in df.columns]    #从预定义的特征列表中，只保留那些在数据表中实际存在的列

        X = df[selected_features].copy()    #从数据表 df 中选取指定的特征列，并创建一个独立的副本，赋值给 X

        #类别特征编码
        for col in CATEGORICAL_FEATURES:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))   #将该列文本转换为数字，并替换原列。X['proto'] = [2, 1, 2, 0, 1]

        print(f"   特征数: {len(selected_features)} (数值{len(NUMERIC_FEATURES)}, 类别{len(CATEGORICAL_FEATURES)})")
        # ---------- 3. 数据划分 ----------
        print(f"\n✂️ [3/5] 数据划分...")

        #创建索引数组
        indices = np.arange(len(X))
        # 获取标签（用于分层抽样）
        y_str = df[TARGET_COLUMN].astype(str)

        #第一步：分出训练集（60%）和临时集（40%）
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.4,  # 临时集占40%
            random_state=RANDOM_STATE, stratify=y_str.values  # 按标签分层
        )

        #第二步：从临时集中分出验证集（50%）和测试集（50%）
        #即各占总数据的20%
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5,  # 测试集占临时集的50%
            random_state=RANDOM_STATE, stratify=y_str.values[temp_idx]  # 按临时集的标签分层
        )

        # 根据索引提取数据
        X_train = X.iloc[train_idx]  #从特征表中取出训练集的行（60%的数据）
        X_val = X.iloc[val_idx]      #从特征表中取出验证集的行（20%的数据）
        X_test = X.iloc[test_idx]    #从特征表中取出测试集的行（20%的数据）

        y_train = y[train_idx]  #从标签表中取出训练集的标签（60%的数据）
        y_val = y[val_idx]      #从标签表中取出验证集的标签（20%的数据）
        y_test = y[test_idx]    #从标签表中取出测试集的标签（20%的数据）

        print(f"   训练集: {len(X_train)} ({len(X_train) / len(X) * 100:.1f}%)")
        print(f"   验证集: {len(X_val)} ({len(X_val) / len(X) * 100:.1f}%)")
        print(f"   测试集: {len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)")
        #标准化
        scaler = StandardScaler()
        X_train[NUMERIC_FEATURES] = scaler.fit_transform(X_train[NUMERIC_FEATURES])
        X_val[NUMERIC_FEATURES] = scaler.transform(X_val[NUMERIC_FEATURES])
        X_test[NUMERIC_FEATURES] = scaler.transform(X_test[NUMERIC_FEATURES])
        # ---------- 4. 训练模型 ----------
        print(f"\n🚀 [4/5] 训练随机森林...")
        print(f"   参数: n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, class_weight={CLASS_WEIGHT}")

        start_time = time.time()            #记录训练开始的时间戳（秒数）
        rf_model = RandomForestClassifier(  #创建随机森林模型对象
            n_estimators=N_ESTIMATORS,      #决策树数量（200棵）
            max_depth=MAX_DEPTH,            #每棵树最大深度（30层）
            class_weight=CLASS_WEIGHT,      #类别权重（'balanced'，自动平衡不平衡数据）
            random_state=RANDOM_STATE,      #随机种子（42，保证结果可复现）
            n_jobs=-1,                      #使用所有CPU核心并行训练（-1表示全部）
            verbose=0                       #不打印训练过程日志（0=静默）
        )
        rf_model.fit(X_train, y_train)      #用训练集训练模型（学习特征和标签的关系）
        train_time = time.time() - start_time  #计算训练耗时 = 结束时间 - 开始时间（秒）

        #验证集评估
        y_val_pred = rf_model.predict(X_val)   #用训练好的模型对验证集进行预测，返回预测的标签
        val_f1 = f1_score(y_val, y_val_pred, average='macro')#计算验证集的宏平均F1分数

        print(f"   ✅ 训练完成! 耗时: {train_time:.2f}s, 验证集F1: {val_f1:.4f}")

        # ---------- 5. 测试评估 ----------
        print(f"\n📊 [5/5] 测试集评估...")

        #完整预测
        y_pred = rf_model.predict(X_test)   #用训练好的模型对测试集进行预测，返回预测结果（预测标签）
        y_prob = rf_model.predict_proba(X_test)  #获取预测概率（用于AUC计算）

        #基础指标
        acc = accuracy_score(y_test, y_pred)    #准确率：预测正确的样本数 / 总样本数
        macro_f1 = f1_score(y_test, y_pred, average='macro')  # 宏平均F1每个类别F1先单独计算，再取平均

        #将标签二值化（用于多分类AUC）
        y_test_bin = label_binarize(y_test, classes=range(len(class_names)))    #二值化（One-Hot）标签矩阵

        #计算AUC-ROC和AUC-PR
        try:
            auc_roc = roc_auc_score(y_test_bin, y_prob, average='macro', multi_class='ovr') #宏平均 ROC-AUC 值（Macro-average ROC-AUC），评估模型整体区分能力
            auc_pr = average_precision_score(y_test_bin, y_prob, average='macro')           #宏平均 PR-AUC 值（Macro-average PR-AUC），评估模型对少数类的检测能力，适用于不平衡数据
        except:
            auc_roc = 0.0
            auc_pr = 0.0

        # ========== 计算多分类FPR/FNR（复用主函数的MetricsCalculator）==========

        #创建类别名称映射字典
        attack_names = {i: name for i, name in enumerate(class_names)}  #0:'backdoor'

        #计算全面指标
        metrics = MetricsCalculator.calculate_all(
            y_test, y_pred, y_prob, attack_names    #真实标签，预测标签，预测概率，类别名称映射字典
        )

        macro_fpr = metrics['macro_fpr']        #宏平均误报率：每个类别FPR先单独计算，再取算术平均（每个类别权重相同）
        macro_fnr = metrics['macro_fnr']        #宏平均漏报率：每个类别FNR先单独计算，再取算术平均（每个类别权重相同）
        weighted_fpr = metrics['weighted_fpr']  #加权平均误报率：按每个类别的样本数加权计算（多数类权重更大）
        weighted_fnr = metrics['weighted_fnr']  #加权平均漏报率：按每个类别的样本数加权计算（多数类权重更大）

        # ========== 二分类视角FPR/FNR（保留）==========
        cm = confusion_matrix(y_test, y_pred)#混淆矩阵
        tn = cm[normal_idx, normal_idx]     #真阴性：正常流量被正确预测为正常
        fp = cm[normal_idx, :].sum() - tn   #假阳性：正常流量被错误预测为攻击（误报）
        fn = cm[:, normal_idx].sum() - tn   #假阴性：攻击流量被错误预测为正常（漏报）
        tp = cm.sum() - tn - fp - fn        #真阳性：攻击流量被正确预测为攻击

        binary_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0   #假阳性率（误报率）
        binary_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0   #假阴性率（漏报率）

        # ========== 打印结果（与主函数格式一致）==========
        print(f"\n{'=' * 70}")
        print("📈 评估结果 (随机森林基线)")
        print(f"{'=' * 70}")
        print(f"🏆 最终 Macro-F1: {macro_f1:.4f}")
        print(f"📊 Accuracy: {acc:.4f}")
        print(f"📈 Mean AUC-ROC: {auc_roc:.4f}")
        print(f"📉 Mean AUC-PR: {auc_pr:.4f}")
        print(f"❌ 宏平均误报率 (Macro-FPR): {macro_fpr:.4f}")
        print(f"⚠️ 宏平均漏报率 (Macro-FNR): {macro_fnr:.4f}")
        print(f"⚖️ 加权误报率: {weighted_fpr:.4f}")
        print(f"⚖️ 加权漏报率: {weighted_fnr:.4f}")
        print(f"{'=' * 70}")
        print(f"🕐 训练时间: {train_time:.2f} 秒")
        print(f"{'=' * 70}")

        #打印二分类视角（参考）
        print(f"\n[二分类视角 - Normal vs Attack]")
        print(f"   误报率 (FPR): {binary_fpr:.4f} ({int(fp)}/{int(fp + tn)})")
        print(f"   漏报率 (FNR): {binary_fnr:.4f} ({int(fn)}/{int(fn + tp)})")

        #每类详细指标
        print(f"\n📋 各类别性能:")
        for i, name in enumerate(class_names):
            mask = ( y_test == i )  #创建布尔掩码：找出测试集中所有真实标签等于当前类别 i 的样本
            if mask.sum() == 0:     #如果该类别的样本数量为0，跳过（不处理空类别）
                continue
            p = precision_score(y_test == i, y_pred == i, zero_division=0)  #计算当前类别的精确率（Precision）
            r = recall_score(y_test == i, y_pred == i, zero_division=0)     #计算当前类别的召回率（Recall）
            f1 = f1_score(y_test == i, y_pred == i, zero_division=0)        #计算当前类别的F1分数
            print(f"  {name:12s}: P={p:.4f}, R={r:.4f}, F1={f1:.4f}, n={mask.sum()}")#打印该类别性能：类别名称、精确率、召回率、F1、样本数

        # ---------- 6. 保存结果 -----------
        # 构建结果字典，用于保存所有实验信息
        results = {
            'timestamp': timestamp,   #实验时间戳（如 "20260414_143025"），用于区分不同次运行
            'model': 'RandomForest',  #模型名称（随机森林）
            'configuration': {                  #模型超参数配置
                'n_estimators': N_ESTIMATORS,   #决策树数量（200棵）
                'max_depth': MAX_DEPTH,         #树的最大深度（30层）
                'class_weight': CLASS_WEIGHT,   #类别权重策略（'balanced'，自动平衡）
                'random_state': RANDOM_STATE,   #随机种子（42，保证可复现）
            },
            'metrics': {                #模型性能指标
                # 基础指标
                'accuracy': float(acc),         #准确率：正确预测的样本数 / 总样本数
                'macro_f1': float(macro_f1),    #宏平均F1：每个类别F1先单独计算，再取平均
                # AUC指标（曲线下面积）
                'auc_roc': float(auc_roc),      #AUC-ROC：ROC曲线下面积，评估整体区分能力（0.5~1.0）
                'auc_pr': float(auc_pr),        #AUC-PR：PR曲线下面积，评估不平衡数据下的表现（0.0~1.0）
                # 多分类视角的误报率/漏报率
                'macro_fpr': float(macro_fpr),  #宏平均误报率：每个类别FPR取平均（每个类别权重相同）
                'macro_fnr': float(macro_fnr),  #宏平均漏报率：每个类别FNR取平均（每个类别权重相同）
                'weighted_fpr': float(weighted_fpr),  #加权平均误报率：按样本数加权（多数类权重更大）
                'weighted_fnr': float(weighted_fnr),  #加权平均漏报率：按样本数加权（多数类权重更大）
                # 二分类视角的误报率/漏报率（正常 vs 攻击）
                'binary_fpr': float(binary_fpr),  #二分类误报率：正常流量被误判为攻击的比例
                'binary_fnr': float(binary_fnr),  #二分类漏报率：攻击流量被漏判为正常的比例
                # 时间指标
                'train_time_sec': float(train_time),  #训练耗时（秒）
            },
            # ========== 类别信息 ==========
            'class_names': list(class_names),  #类别名称列表，如 ['backdoor', 'ddos', 'normal', ...]
            'normal_class_index': int(normal_idx),  #正常类别在class_names中的索引位置（用于计算FPR/FNR）
            # ========== 特征信息 ==========
            'feature_names': selected_features  #用于训练的特征列名列表（共11个）
        }

        run_dir = save_results(results, y_test, y_pred, class_names, timestamp)

        # ----------- 7. 结论 ----------
        print(f"\n{'=' * 70}")
        print("💡 基线结论")
        print(f"{'=' * 70}")
        print(f"强基线性能: Macro-F1={macro_f1:.4f}, 宏平均FPR={macro_fpr:.4f}")

        print(f"结果保存: {run_dir}")
        print(f"{'=' * 70}")

        # 保存模型
        model_path = os.path.join(run_dir, 'model.pkl')
        #将模型和相关配置保存为文件
        joblib.dump({
            'model': rf_model,              #训练好的随机森林模型
            'label_encoder': le_target,     #标签编码器（用于数字→文本转换）
            'config': {         #模型配置信息
                'normal_idx': normal_idx,           #正常类别的索引
                'class_names': list(class_names),   #所有类别名称列表
                'features': selected_features       #使用的特征列名
            }
        }, model_path)
        print(f"\n💾 模型已保存: {model_path}")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()