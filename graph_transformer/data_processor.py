import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
from typing import Dict, List, Tuple
from config import Config

warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = str(42)

# ================= 数据处理器 =================
class DataProcessor:
    """数据处理类"""

    def __init__(self, config: Config):
        self.config = config                #保存配置对象（包含所有参数设置）
        self.label_encoders = {}            #存储所有类别特征的编码器（字典，键是列名）
        self.scaler = StandardScaler()      #标准化器，用于数值特征的标准化（均值0，方差1）
        self.attack_encoder = LabelEncoder()#攻击类型的编码器，将攻击名称转为数字

    def load_data(self) -> pd.DataFrame:#函数返回一个 pandas DataFrame 对象
        """加载数据"""
        print("\n[1/4] 加载数据...")
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"数据文件不存在: {self.config.data_path}")

        df = pd.read_csv(self.config.data_path)
        print(f"数据形状: {df.shape}")
        return df

    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray,np.ndarray, np.ndarray,Dict[int, str], List[str]]:
        """完整的预处理流程"""
        target_col = 'type' if 'type' in df.columns else 'label'    #确定目标列名（可能是'type'或'label'）

        # 划分数据集
        indices = np.arange(len(df))    #创建索引数组[0,1,2,...,n-1]（所有数据的行号）
        temp_target = df[target_col].astype(str)    #将目标列转为字符串，用于分层抽样

        train_idx, temp_idx = train_test_split(
            indices, test_size=0.4, random_state=self.config.random_seed,
            stratify=temp_target.values         #按标签分层抽样，保持类别比例
        )#把全部数据集索引划分，临时集占40%,返回两个索引数组

        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=self.config.random_seed,
            stratify=temp_target.values[temp_idx]  #只用临时集的标签分层
        )#把临时集划分，测试集占临时集的50%

        train_df = df.iloc[train_idx].copy()    #提取训练集数据
        val_df = df.iloc[val_idx].copy()        #提取验证集数据
        test_df = df.iloc[test_idx].copy()      #提取测试集数据

        #从数据集中筛选出用于模型训练的特征列
        exclude_cols = ['src_ip', 'dst_ip', target_col]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        #将feature_cols中的文本类别特征（如协议类型、标志位等）转换为数字
        train_df, val_df, test_df = self._encode_categorical_features(
            train_df, val_df, test_df, feature_cols)

        #数值型特征进行标准化处理
        train_df, val_df, test_df = self._standardize_numeric_features(
            train_df, val_df, test_df, feature_cols)

        #将目标列（攻击类型）从文本转换为数字标签，并返回转换后的数据集和类别名称映射
        train_df, val_df, test_df, attack_names = self._encode_target(
            train_df, val_df, test_df, target_col)      #返回的映射字典attack_names

        #合并数据集，并添加一列标记每条数据属于哪个集合，eg:train_df['_split'] = 'train'..
        final_df = self._merge_datasets(train_df, val_df, test_df)

        #创建布尔掩码（Boolean Mask），用于标记哪些数据属于训练集、验证集和测试集
        train_mask = (final_df['_split'] == 'train').values
        val_mask = (final_df['_split'] == 'val').values
        test_mask = (final_df['_split'] == 'test').values   #三个布尔数组

        #从布尔掩码中提取出满足条件（为TRUE）的索引位置
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]

        #打印训练集中每个攻击类型的样本数量分布
        self._print_class_distribution(final_df[train_mask], attack_names)

        return final_df, train_idx, val_idx, test_idx, attack_names, feature_cols

    def _encode_categorical_features(self, train_df, val_df, test_df, feature_cols):
        """编码类别特征"""
        for col in feature_cols:
            if train_df[col].dtype == 'object':     #如果是文本类型，进行编码
                le = LabelEncoder()     #文本到数字的映射规则

                train_values = train_df[col].fillna('Unknown').astype(str).unique()     #获取训练集某一列的所有唯一值，并将缺失值替换为'Unknown'后统一转为字符串格式。
                all_possible_values = list(train_values)    #将NumPy数组转换为列表

                if 'Unknown' not in all_possible_values:
                    all_possible_values.append('Unknown')   #添加'Unknown'元素

                le.fit(all_possible_values)     #训练标签编码器，学习文本到数字的映射关系
                self.label_encoders[col] = le

                for df_subset in [train_df, val_df, test_df]:
                    df_subset[col] = df_subset[col].fillna('Unknown').astype(str)       #填充缺失值+转字符串
                    df_subset[col] = df_subset[col].apply(lambda x: x if x in le.classes_ else 'Unknown')   #替换未知值
                    df_subset[col] = le.transform(df_subset[col])           #转换为数字

            else:       #如果不是文本（数值类型），只填充缺失值为0
                for df_subset in [train_df, val_df, test_df]:
                    df_subset[col] = df_subset[col].fillna(0)

        return train_df, val_df, test_df

    def _standardize_numeric_features(self, train_df, val_df, test_df, feature_cols):
        """标准化数值特征"""
        numeric_cols = [c for c in feature_cols if train_df[c].dtype != 'object']
        if numeric_cols:
            train_df[numeric_cols] = self.scaler.fit_transform(train_df[numeric_cols])  #计算训练集该列的均值和标准差,用计算出的均值和标准差进行标准化
            val_df[numeric_cols] = self.scaler.transform(val_df[numeric_cols])      #用计算出的均值和标准差进行标准化
            test_df[numeric_cols] = self.scaler.transform(test_df[numeric_cols])    #用计算出的均值和标准差进行标准化

        return train_df, val_df, test_df

    def _encode_target(self, train_df, val_df, test_df, target_col):
        """编码目标列"""
        train_targets = train_df[target_col].astype(str).unique()
        self.attack_encoder.fit(train_targets)

        train_df['attack_type'] = self.attack_encoder.transform(
            train_df[target_col].astype(str)
        )

        most_common = train_targets[0]
        for df_subset in [val_df, test_df]:
            targets = df_subset[target_col].astype(str)
            df_subset['attack_type'] = targets.apply(
                lambda x: self.attack_encoder.transform([x])[0]
                if x in self.attack_encoder.classes_
                else self.attack_encoder.transform([most_common])[0]
            )

        attack_names = {i: name for i, name in enumerate(self.attack_encoder.classes_)}
        return train_df, val_df, test_df, attack_names

    def _merge_datasets(self, train_df, val_df, test_df):
        """合并数据集"""
        train_df['_split'] = 'train'        #新增一列 _split
        val_df['_split'] = 'val'
        test_df['_split'] = 'test'

        return pd.concat([train_df, val_df, test_df], ignore_index=True)   #忽略原来的索引，重新生成新的连续索引

    def _print_class_distribution(self, train_df, attack_names):
        """打印类别分布"""
        print(f"\n📊 训练集类别分布:")
        for i in range(len(attack_names)):
            count = (train_df['attack_type'] == i).sum()
            print(f"   {attack_names[i]}: {count}")