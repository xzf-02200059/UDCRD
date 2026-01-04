#NADEEF,dBoost,outlier(std,iqr,iso),mv,RandomForest,FAHES,KATARA,Raha(dBoost+KATARA)
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import re
from scipy.stats import gaussian_kde
from collections import defaultdict
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import logging
from itertools import combinations

# 加载数据
dirty_data_path = 'retailSales_80%.csv'
clean_data_path = 'retailSales_clean.csv'
knowledge_base_path = 'global_knowledge_base.json'


def clean_fields_to_replace(data):
    """
    清洗 `fields_to_replace` 列，确保格式为：
    - 如果为空，则为 []
    - 如果有一个元素，则为 ['thickness']
    - 如果有多个元素，则为 ['chord_length', 'velocity', 'sound_pressure_level']
    """
    for index, row in data.iterrows():
        fields_str = row['fields_to_replace']

        # 如果是空值或空字符串，替换为 []
        if not fields_str or fields_str == '[""]' or fields_str == '""':
            data.at[index, 'fields_to_replace'] = []
        else:
            # 如果是字符串类型
            if isinstance(fields_str, str):
                # 去掉外层的引号并按逗号分隔
                fields_str = fields_str.strip("[]").replace('"', '')

                # 分割字符串并返回为列表
                fields_list = [field.strip() for field in fields_str.split(',')]
                data.at[index, 'fields_to_replace'] = fields_list
            elif isinstance(fields_str, list):
                # 如果已经是列表类型，直接返回该列表
                data.at[index, 'fields_to_replace'] = fields_str

    return data

# 定义约束检测函数
# 外部定义数据集约束和函数依赖
dataset_constraints = {
    "hospital": {
        "functions": [
            ["City", "ZipCode"],
            ["City", "CountyName"],
            ["ZipCode", "City"],
            ["ZipCode", "State"],
            ["ZipCode", "CountyName"],
            ["CountyName", "State"]
        ],
        "patterns": [
            ["index", "^[\\d]+$", "ONM"],
            ["ProviderNumber", "^[\\d]+$", "ONM"],
            ["ZipCode", "^[\\d]{5}$", "ONM"],
            ["State", "^[a-z]{2}$", "ONM"],
            ["PhoneNumber", "^[\\d]+$", "ONM"]
        ]
    },
    "flight": {
        "functions": [
            ["flight", "act_dep_time"], ["flight", "sched_arr_time"], ["flight", "act_arr_time"],["flight", "sched_dep_time"]
        ],
        "patterns": [
        ]
    },
    "beers": {
        "functions": [
            ["brewery_id", "brewery_name"], ["brewery_id", "city"], ["brewery_id", "state"]
        ],
        "patterns": [
            ["state", "^[A-Z]{2}$", "ONM"], ["brewery_id", "^[\d]+$", "ONM"]
        ]
    }
}
def nadeef_error_detection(clean_data_path, dirty_data_path, dataset_name):
    """
    使用内置的函数依赖和约束进行错误检测，并将结果保存到文件。
    返回:
    - 包含权重列的 Pandas DataFrame
    """
    try:
        # 加载数据
        sf_clean = pd.read_csv(clean_data_path)
        sf_dirty = pd.read_csv(dirty_data_path)
        output_path = 'NADEEF_output.csv'

        # 根据数据集名称获取对应的约束和函数依赖
        constraints = dataset_constraints.get(dataset_name, {})
        functions = constraints.get("functions", [])
        patterns = constraints.get("patterns", [])

        # 预处理数据类型
        numeric_columns = sf_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
        for col in numeric_columns:
            sf_dirty[col] = pd.to_numeric(sf_dirty[col], errors='coerce')

        def check_functional_dependencies(row, functions):
            fields_to_replace = []
            if not functions:
                return fields_to_replace
            for fd in functions:
                l_attribute, r_attribute = fd
                # 检查左属性和右属性是否满足函数依赖
                # 如果左属性有值，但右属性为空或不符合函数依赖，则标记右属性
                if pd.notna(row.get(l_attribute)):
                    # 检查右属性是否为空
                    if pd.isna(row.get(r_attribute)):
                        fields_to_replace.append(r_attribute)
                    # 检查函数依赖是否成立（这里简化处理，实际可能需要更复杂的逻辑）
                    # 例如，可以检查在干净数据集中，相同的左属性值是否总是对应相同的右属性值
            return fields_to_replace

        def check_patterns(row, patterns):
            fields_to_replace = []
            if not patterns:
                return fields_to_replace
            for pattern in patterns:
                attribute, regex_pattern, opcode = pattern
                value = row.get(attribute)
                if pd.notna(value) and isinstance(value, str):
                    if opcode == "ONM" and not re.match(regex_pattern, value):
                        fields_to_replace.append(attribute)
            return fields_to_replace

        def detect_and_replace_errors(row):
            """
            检测单行数据是否违反预定义的函数依赖和约束
            """
            fields_to_replace = []

            # 1. 检查函数依赖
            if functions:
                fields_to_replace.extend(check_functional_dependencies(row, functions))

            # 2. 检查模式约束
            if patterns:
                fields_to_replace.extend(check_patterns(row, patterns))

            # 3. 计算权重：如果有字段需要替换，则权重为1，否则为0
            weight = 1 if fields_to_replace else 0

            return list(set(fields_to_replace)), weight

        # 应用错误检测
        results = [detect_and_replace_errors(row) for _, row in sf_dirty.iterrows()]
        sf_dirty['fields_to_replace'], sf_dirty['weight'] = zip(*results)

        # 保存结果
        sf_dirty.to_csv(output_path, index=False)
        print(f"错误检测完成，结果已保存到 {output_path}")
        return sf_dirty

    except Exception as e:
        print(f"NADEEF 错误检测失败: {str(e)}")
        return None


def dboost_error_detection(clean_data_path, dirty_data_path):
    # 加载数据
    clean_data = pd.read_csv(clean_data_path)
    dirty_data = pd.read_csv(dirty_data_path)
    output_path = 'dBoost_output.csv'

    # 初始化权重列
    dirty_data['weight'] = 0
    dirty_data['fields_to_replace'] = [[] for _ in range(len(dirty_data))]

    # 检测数值型和分类型列
    numeric_columns = clean_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = clean_data.select_dtypes(include=['object']).columns.tolist()

    # 数值型列：使用高斯混合模型（GMM）检测离群值
    for column in numeric_columns:
        try:
            # 转换为数值类型并处理 NaN 值
            clean_column_data = pd.to_numeric(clean_data[column], errors='coerce').dropna().values.reshape(-1, 1)
            dirty_column_data = pd.to_numeric(dirty_data[column], errors='coerce').fillna(
                clean_column_data.mean()).values.reshape(-1, 1)

            # 使用 GridSearch 寻找最佳组件数
            gmm = GaussianMixture()
            max_components = min(5, len(set(clean_column_data.flatten())))  # 不超过唯一点的数量
            param_grid = {'n_components': range(1, max_components + 1)}  # 寻找 1 到 5 个高斯分量
            grid_search = GridSearchCV(gmm, param_grid, cv=3)
            grid_search.fit(clean_column_data)

            # 使用最佳模型进行预测
            best_gmm = grid_search.best_estimator_
            log_probs = best_gmm.score_samples(dirty_column_data)

            # 设定阈值，标记低概率值为异常
            threshold = np.percentile(log_probs, 5)  # 低于第5百分位的样本视为异常
            outlier_mask = log_probs < threshold

            # 更新权重和 fields_to_replace 列
            dirty_data.loc[outlier_mask, 'weight'] = 1
            dirty_data.loc[outlier_mask, 'fields_to_replace'] = dirty_data.loc[outlier_mask, 'fields_to_replace'].apply(
                lambda x: x + [column]
            )
        except Exception as e:
            print(f"数值列 {column} 的异常值检测失败: {e}")

    # 分类型列：使用 Kernel Density Estimation (KDE) 检测异常
    for column in categorical_columns:
        try:
            # 将类别转换为数值
            clean_column_data = pd.get_dummies(clean_data[column].dropna()).values

            # 使用 KDE 建模
            kde = KernelDensity(kernel='gaussian')
            param_grid = {'bandwidth': np.logspace(-1, 1, 10)}  # 网格搜索带宽参数
            grid_search = GridSearchCV(kde, param_grid, cv=3)
            grid_search.fit(clean_column_data)

            # 使用最佳模型进行预测
            best_kde = grid_search.best_estimator_
            dirty_column_data = pd.get_dummies(dirty_data[column]).reindex(
                columns=pd.get_dummies(clean_data[column]).columns, fill_value=0).values
            log_probs = best_kde.score_samples(dirty_column_data)

            # 标记低概率值为异常
            threshold = np.percentile(log_probs, 5)  # 低于第5百分位的样本视为异常
            outlier_mask = log_probs < threshold

            # 更新权重和 fields_to_replace 列
            dirty_data.loc[outlier_mask, 'weight'] = 1
            dirty_data.loc[outlier_mask, 'fields_to_replace'] = dirty_data.loc[outlier_mask, 'fields_to_replace'].apply(
                lambda x: x + [column]
            )
        except Exception as e:
            print(f"分类列 {column} 的异常值检测失败: {e}")

    # 保存检测结果到文件
    dirty_data.to_csv(output_path, index=False)
    print(f"错误检测完成，结果已保存到 {output_path}")

    return dirty_data

def standard_deviation_outlier_detection(data, clean_data, columns, n=3):
    """
    使用标准差方法检测离群值，并标记错误单元格。
    """
    for column in columns:
        # 确保列为数值型
        data[column] = pd.to_numeric(data[column], errors='coerce')
        clean_data[column] = pd.to_numeric(clean_data[column], errors='coerce')

        # 标记包含无效值的行
        data['fields_to_replace'] = data.apply(
            lambda row: row['fields_to_replace'] + [column] if pd.isna(row[column]) else row['fields_to_replace'], axis=1
        )

        # 计算均值和标准差（忽略 NaN）
        mean = clean_data[column].mean()
        std = clean_data[column].std()

        # 检测离群值（忽略 NaN）
        data['fields_to_replace'] = data.apply(
            lambda row: row['fields_to_replace'] + [column]
            if not pd.isna(row[column]) and (row[column] < mean - n * std or row[column] > mean + n * std)
            else row['fields_to_replace'], axis=1
        )
    return data

def iqr_outlier_detection(data, clean_data, columns, k=1.5):
    """
    使用四分位数间距 (IQR) 方法检测离群值，并标记错误单元格。
    """
    for column in columns:
        # 确保列为数值型
        data[column] = pd.to_numeric(data[column], errors='coerce')
        clean_data[column] = pd.to_numeric(clean_data[column], errors='coerce')

        # 标记包含无效值的行
        data['fields_to_replace'] = data.apply(
            lambda row: row['fields_to_replace'] + [column] if pd.isna(row[column]) else row['fields_to_replace'], axis=1
        )

        # 计算四分位数和 IQR（忽略 NaN）
        Q1 = clean_data[column].quantile(0.25)
        Q3 = clean_data[column].quantile(0.75)
        IQR = Q3 - Q1

        # 检测离群值（忽略 NaN）
        data['fields_to_replace'] = data.apply(
            lambda row: row['fields_to_replace'] + [column]
            if not pd.isna(row[column]) and (row[column] < Q1 - k * IQR or row[column] > Q3 + k * IQR)
            else row['fields_to_replace'], axis=1
        )
    return data

def isolation_forest_outlier_detection(data, clean_data, columns, contamination=0.05):
    """
    使用孤立森林方法检测离群值，并标记错误单元格。
    """
    for column in columns:
        # 确保列为数值型
        data[column] = pd.to_numeric(data[column], errors='coerce')
        clean_data[column] = pd.to_numeric(clean_data[column], errors='coerce')

        # 标记包含无效值的行
        data['fields_to_replace'] = data.apply(
            lambda row: row['fields_to_replace'] + [column] if pd.isna(row[column]) else row['fields_to_replace'], axis=1
        )

        # 对非 NaN 值训练孤立森林
        mask = ~data[column].isna()
        if mask.sum() > 0:  # 如果非 NaN 值数量大于 0
            model = IsolationForest(contamination=contamination, random_state=42)
            model.fit(clean_data.loc[mask, column].values.reshape(-1, 1))

            # 检测离群值
            data['fields_to_replace'] = data.apply(
                lambda row: row['fields_to_replace'] + [column]
                if mask[row.name] and model.predict([[row[column]]])[0] == -1
                else row['fields_to_replace'], axis=1
            )
    return data

def outlier_error_detection(clean_data_path, dirty_data_path):
    """
    使用标准差、四分位数间距、孤立森林检测错误，并输出包含 fields_to_replace 和 weight 的数据。
    """
    # 加载数据
    clean_data = pd.read_csv(clean_data_path)
    dirty_data = pd.read_csv(dirty_data_path)
    output_path = 'outlier_output.csv'

    # 数值型列
    numeric_columns = clean_data.select_dtypes(include=[np.number]).columns.tolist()

    # 初始化 fields_to_replace 列
    dirty_data['fields_to_replace'] = [[] for _ in range(len(dirty_data))]

    # 标准差检测
    dirty_data = standard_deviation_outlier_detection(dirty_data, clean_data, numeric_columns, n=3)

    # 四分位数间距检测
    dirty_data = iqr_outlier_detection(dirty_data, clean_data, numeric_columns, k=1.5)

    # 孤立森林检测
    dirty_data = isolation_forest_outlier_detection(dirty_data, clean_data, numeric_columns, contamination=0.05)

    # 计算总体权重列（如果 fields_to_replace 不为空，则 weight = 1）
    dirty_data['weight'] = dirty_data['fields_to_replace'].apply(lambda x: 1 if len(x) > 0 else 0)

    # 保存检测结果
    dirty_data.to_csv(output_path, index=False)
    print(f"错误检测完成，结果已保存到 {output_path}")

    return dirty_data


def mv_error_detection(dirty_data_path):
    # 读取脏数据
    dirty_data = pd.read_csv(dirty_data_path)
    output_path = 'mv_output.csv'

    # 初始化 fields_to_replace 列
    dirty_data['fields_to_replace'] = dirty_data.apply(
        lambda row: [col for col in dirty_data.columns if pd.isna(row[col])], axis=1
    )

    # 初始化权重列，根据 fields_to_replace 列是否为空设置权重
    dirty_data['weight'] = dirty_data['fields_to_replace'].apply(lambda x: 1 if len(x) > 0 else 0)

    # 检测每一行是否有缺失值
    #dirty_data['weight'] = dirty_data.isnull().any(axis=1).astype(int)

    # 将检测后的数据保存到文件
    dirty_data.to_csv(output_path, index=False)
    print(f"缺失值检测完成，结果已保存到 {output_path}")

    return dirty_data


def RandomForest_error_detection(dirty_data_path, clean_data_path):
    # 加载数据
    dirty_data = pd.read_csv(dirty_data_path)
    clean_data = pd.read_csv(clean_data_path)
    output_path = 'RandomForest_output.csv'

    # 确保所有列都可以正确处理数值型数据
    for col in dirty_data.columns:
        dirty_data[col] = pd.to_numeric(dirty_data[col], errors='coerce')

    # 生成错误矩阵
    error_matrix = dirty_data.ne(clean_data)  # 布尔矩阵，True表示对应单元格有错误

    # 特征和标签准备
    features = []
    labels = []
    for row_idx, row in dirty_data.iterrows():
        for col_idx, col in enumerate(dirty_data.columns):
            cell_value = row[col]
            if not pd.isna(cell_value):  # 确保值不是 NaN
                features.append([
                    cell_value,  # 数值值
                    col_idx,  # 列索引
                    dirty_data[col].mean(),  # 列均值
                    dirty_data[col].std(),  # 列标准差
                    0 if isinstance(cell_value, (int, float)) else 1  # 是否为非数值
                ])
            else:  # 对于 NaN 值
                features.append([0, col_idx, 0, 0, 1])
            labels.append(int(error_matrix.iloc[row_idx, col_idx]))  # 错误矩阵的值作为标签

    # 转换为NumPy数组
    features = np.array(features)
    labels = np.array(labels)

    # 数据平衡处理：使用随机森林分类器
    model = RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42)
    model.fit(features, labels)

    # 准备脏数据特征用于预测
    dirty_features = []
    for row_idx, row in dirty_data.iterrows():
        for col_idx, col in enumerate(dirty_data.columns):
            cell_value = row[col]
            if not pd.isna(cell_value):
                dirty_features.append([
                    cell_value,
                    col_idx,
                    dirty_data[col].mean(),
                    dirty_data[col].std(),
                    0 if isinstance(cell_value, (int, float)) else 1
                ])
            else:
                dirty_features.append([0, col_idx, 0, 0, 1])
    dirty_features = np.array(dirty_features)

    # 用模型预测错误
    predictions = model.predict(dirty_features)  # 预测错误标记

    # 重新组织预测结果，生成权重列和辅助列
    weights = []
    fields_to_replace = []
    idx = 0
    for row_idx in range(dirty_data.shape[0]):
        row_errors = []
        for col_idx in range(dirty_data.shape[1]):
            if predictions[idx] == 1:  # 如果预测为错误
                row_errors.append(dirty_data.columns[col_idx])
            idx += 1
        weights.append(1 if row_errors else 0)  # 如果有任何错误，权重为1
        fields_to_replace.append(f'["{",".join(row_errors)}"]')

        # 将结果写回原始数据
    dirty_data["weight"] = weights
    dirty_data["fields_to_replace"] = fields_to_replace

    # 假设 `dirty_data` 是你加载的数据
    dirty_data = clean_fields_to_replace(dirty_data)

    # 保存结果
    dirty_data.to_csv(output_path, index=False)
    print(f"检测后的数据已保存到 {output_path}")
    return dirty_data

# def RandomForest_error_detection(dirty_data_path, clean_data_path):
#     # 加载数据
#     dirty_data = pd.read_csv(dirty_data_path)
#     clean_data = pd.read_csv(clean_data_path)
#     output_path = 'RandomForest_output.csv'
#
#     # 处理混合型数据：区分数值型和分类数据
#     numeric_cols = dirty_data.select_dtypes(include=[np.number]).columns
#     categorical_cols = dirty_data.select_dtypes(include=[object]).columns
#
#     # 填充缺失值
#     dirty_data[numeric_cols] = dirty_data[numeric_cols].fillna('Unknown')
#     clean_data[numeric_cols] = clean_data[numeric_cols].fillna('Unknown')
#     dirty_data[categorical_cols] = dirty_data[categorical_cols].fillna('Unknown')
#     clean_data[categorical_cols] = clean_data[categorical_cols].fillna('Unknown')
#
#     # 对分类型数据进行 One-Hot 编码
#     ohe = OneHotEncoder(sparse_output=False, drop='first',handle_unknown='ignore')
#     categorical_data = ohe.fit_transform(dirty_data[categorical_cols])
#
#     # 将编码后的数据拼接回原数据中
#     encoded_df = pd.DataFrame(categorical_data, columns=ohe.get_feature_names_out(categorical_cols))
#     dirty_data_encoded = pd.concat([dirty_data[numeric_cols], encoded_df], axis=1)
#
#     # 确保所有列都可以正确处理数值型数据
#     for col in dirty_data_encoded.columns:
#         dirty_data_encoded[col] = pd.to_numeric(dirty_data_encoded[col], errors='coerce')
#
#     # 生成错误矩阵
#     clean_data_encoded = pd.concat([clean_data[numeric_cols], ohe.transform(clean_data[categorical_cols])], axis=1)
#     error_matrix = dirty_data_encoded.ne(clean_data_encoded)  # 布尔矩阵，True表示对应单元格有错误
#
#     # 特征和标签准备
#     features = []
#     labels = []
#     for row_idx, row in dirty_data_encoded.iterrows():
#         for col_idx, col in enumerate(dirty_data_encoded.columns):
#             cell_value = row[col]
#             if not pd.isna(cell_value):  # 确保值不是 NaN
#                 # 对于数值型和分类数据都有处理
#                 features.append([
#                     cell_value,  # 数值值或编码后的值
#                     col_idx,  # 列索引
#                     dirty_data_encoded[col].mean(),  # 列均值
#                     dirty_data_encoded[col].std(),  # 列标准差
#                     0 if isinstance(cell_value, (int, float)) else 1  # 是否为分类型数据
#                 ])
#             else:  # 对于 NaN 值
#                 features.append([0, col_idx, 0, 0, 1])  # 分类型数据也能处理
#             labels.append(int(error_matrix.iloc[row_idx, col_idx]))  # 错误矩阵的值作为标签
#
#     # 转换为NumPy数组
#     features = np.array(features)
#     labels = np.array(labels)
#
#     # 数据平衡处理：使用随机森林分类器
#     model = RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42)
#     model.fit(features, labels)
#
#     # 准备脏数据特征用于预测
#     dirty_features = []
#     for row_idx, row in dirty_data_encoded.iterrows():
#         for col_idx, col in enumerate(dirty_data_encoded.columns):
#             cell_value = row[col]
#             if not pd.isna(cell_value):
#                 dirty_features.append([
#                     cell_value,
#                     col_idx,
#                     dirty_data_encoded[col].mean(),
#                     dirty_data_encoded[col].std(),
#                     0 if isinstance(cell_value, (int, float)) else 1
#                 ])
#             else:
#                 dirty_features.append([0, col_idx, 0, 0, 1])
#     dirty_features = np.array(dirty_features)
#
#     # 用模型预测错误
#     predictions = model.predict(dirty_features)  # 预测错误标记
#
#     # 重新组织预测结果，生成权重列和辅助列
#     weights = []
#     fields_to_replace = []
#     idx = 0
#     for row_idx in range(dirty_data.shape[0]):
#         row_errors = []
#         for col_idx in range(dirty_data.shape[1]):
#             if predictions[idx] == 1:  # 如果预测为错误
#                 row_errors.append(dirty_data.columns[col_idx])
#             idx += 1
#         weights.append(1 if row_errors else 0)  # 如果有任何错误，权重为1
#         fields_to_replace.append(f'["{",".join(row_errors)}"]')
#
#     # 将结果写回原始数据
#     dirty_data["weight"] = weights
#     dirty_data["fields_to_replace"] = fields_to_replace
#
#     # 假设 `dirty_data` 是你加载的数据
#     dirty_data = clean_fields_to_replace(dirty_data)
#
#     # 保存结果
#     dirty_data.to_csv(output_path, index=False)
#     print(f"检测后的数据已保存到 {output_path}")
#     return dirty_data

# class FAHES:
#     def __init__(self, data):
#         """
#         初始化FAHES类
#         :param data: 输入数据，Pandas DataFrame格式
#         """
#         self.data = data
#         self.dmv_results = defaultdict(set)  # 使用集合避免重复记录
#         #self.dmv_results = {}
#
#     def detect_syntactical_dmvs(self, column):
#         """
#         使用语法规则检测伪装缺失值
#         :param column: 待检测的列名
#         """
#         values = self.data[column].astype(str)
#         common_patterns = ['^\\?$', '^-$', '^NA$', '^N/A$', '^NULL$', '^0$', '^00-00-0000$', '^1111111111$',
#                            '^9999999999$']
#         pattern = re.compile('|'.join(common_patterns))
#
#         dmvs = {value for value in values if pattern.match(value.strip())}
#         self.dmv_results[column].update(dmvs)
#         #self.dmv_results[column] = list(set(dmvs))
#
#     def detect_outlier_dmvskde(self, column):
#         """
#         使用动态密度函数检测异常值（适用于数值列）
#         :param column: 待检测的列名
#         """
#         try:
#             # 转换为数值类型，非数值和空值将被设置为NaN
#             numeric_values = pd.to_numeric(self.data[column], errors='coerce')
#
#             # 保存原始数据长度
#             original_length = len(numeric_values)
#             valid_values = numeric_values.dropna()  # 去除NaN，保留有效数值
#
#             if len(valid_values) < 2:  # 如果有效数据量太少，跳过检测
#                 return
#
#             # 计算高斯核密度估计
#             density = gaussian_kde(valid_values)
#             density_values = density(valid_values)
#             threshold = np.percentile(density_values, 5)  # 设置密度阈值（下5%为异常值）
#
#             # 创建与原始数据对齐的布尔掩码
#             outlier_mask = pd.Series(False, index=numeric_values.index)
#             outlier_mask[valid_values.index] = density_values < threshold
#
#             outliers = numeric_values[outlier_mask < threshold]
#
#             self.dmv_results[column].update(outliers)
#         except Exception as e:
#             print(f"列 {column} 的异常值检测失败: {e}")
#
#     def detect_repeated_dmvs(self, column):
#         """
#         使用使用隐藏时间模式算法/重复模式检测伪装缺失值
#         :param column: 待检测的列名
#         """
#         values = self.data[column].astype(str)
#         value_counts = Counter(values)
#         total_rows = len(self.data)
#         dmvs = {value for value, count in value_counts.items() if count > total_rows * 0.1}
#         self.dmv_results[column].update(dmvs)
#         #self.dmv_results[column] = self.dmv_results.get(column, []) + dmvs
#
#     def detect_random_missing_dmvs_fast(self, column):
#         """
#         使用Fast DiMaC方法检测随机缺失值 (MAR DMVs)
#         :param column: 待检测的列名
#         """
#         # 移除单一值的列
#         if self.data[column].nunique() == len(self.data) or self.data[column].nunique() == 1:
#             return
#             # 移除单一值的列
#         if self.data[column].nunique() == len(self.data) or self.data[column].nunique() == 2:
#             return
#
#         # 构建索引
#         index = defaultdict(list)
#         for idx, value in enumerate(self.data[column]):
#             index[value].append(idx)
#
#         # 计算每个值的相关性
#         table_size = len(self.data)
#         scores = {}
#         for value, indices in index.items():
#             # P(T_Ai="v")，即子集的频率
#             prob_value = len(indices) / table_size
#
#             # P(T_A)，即其他所有值的概率乘积
#             other_values = [v for v in index.keys() if v != value]
#             prob_others = np.prod([len(index[v]) / table_size for v in other_values if len(index[v]) > 0])
#
#             # 相关性公式
#             correlation = prob_value / prob_others if prob_others > 0 else 0
#
#             # 得分公式 S
#             score = sum(
#                 [
#                     (len(index[v]) / table_size) / (1 + abs(correlation - prob_value))
#                     for v in other_values
#                 ]
#             )
#             scores[value] = score
#
#         # 筛选得分较高的值作为DMVs
#         threshold = np.percentile(list(scores.values()), 95)  # 取得分最高的5%值
#         dmvs = {value for value, score in scores.items() if score >= threshold}
#         self.dmv_results[column].update(dmvs)
#         #self.dmv_results[column] = self.dmv_results.get(column, []) + dmvs
#
#     def run_detection(self):
#         """
#         运行所有模块检测伪装缺失值
#         """
#         for column in self.data.columns:
#             self.detect_syntactical_dmvs(column)
#             self.detect_outlier_dmvskde(column)
#             self.detect_repeated_dmvs(column)
#             self.detect_random_missing_dmvs_fast(column)
#
#         # if not self.dmv_results:
#         #     print("Warning: No DMVs detected.")
#         return self.dmv_results
#
#
# def FAHES_error_detection(dirty_data_path):
#     """
#     数据处理流程
#     :param input_file: 输入文件路径
#     :param output_file: 输出文件路径
#     """
#     # 加载数据
#     df = pd.read_csv(dirty_data_path)
#     output_path = 'FAHES_output.csv'
#
#     # 初始化FAHES算法
#     fahes = FAHES(df)
#     dmv_results = fahes.run_detection()
#
#     # 添加fields_to_replace和weight列
#     def identify_errors(row):
#         errors = []
#         for col in df.columns:
#             if col in dmv_results and row[col] in dmv_results[col]:
#                 errors.append(col)
#         return errors
#
#     df['fields_to_replace'] = df.apply(identify_errors, axis=1)
#     df['weight'] = df['fields_to_replace'].apply(lambda x: 1 if x else 0)
#     #df['fields_to_replace'] = df['fields_to_replace'].apply(lambda x: ', '.join(x))
#     df['fields_to_replace'] = df['fields_to_replace'].apply(lambda x: f'["{", ".join(x)}"]')
#     #fields_to_replace.append(f'["{",".join(row_errors)}"]')
#
#     # 假设 `dirty_data` 是你加载的数据
#     df = clean_fields_to_replace(df)
#     #print(type(df['fields_to_replace']))
#
#     # 保存处理后的数据
#     df.to_csv(output_path, index=False)
#     print(f"处理后的数据已保存到 {output_path}")
#     return df
# 配置日志输出
#logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
class FAHES:
    def __init__(self, data):
        """
        初始化FAHES类
        :param data: 输入数据，Pandas DataFrame格式
        """
        self.data = data
        self.dmv_results = defaultdict(set)  # 使用集合避免重复记录

    def detect_syntactical_dmvs(self, column):
        """
        使用语法规则检测伪装缺失值
        目前采用预定义的常见模式，后续可扩展为自动生成模式
        :param column: 待检测的列名
        """
        values = self.data[column].astype(str)
        common_patterns = [
            r'^\?$', r'^-$', r'^NA$', r'^N/A$', r'^NULL$', r'^0$',
            r'^00-00-0000$', r'^1111111111$', r'^9999999999$'
        ]
        pattern = re.compile('|'.join(common_patterns), re.IGNORECASE)

        dmvs = {value.strip() for value in values if pattern.match(value.strip())}
        if dmvs:
            logging.info(f"列 {column} 语法检测到DMVs：{dmvs}")
        self.dmv_results[column].update(dmvs)

    def detect_outlier_dmvskde(self, column):
        """
        使用动态密度函数检测异常值（适用于数值列）
        改进点：在计算密度时排除空值，并通过密度低于阈值判定异常
        :param column: 待检测的列名
        """
        try:
            # 转换为数值类型，非数值和空值设为NaN
            numeric_values = pd.to_numeric(self.data[column], errors='coerce')
            valid_values = numeric_values.dropna()

            if len(valid_values) < 2:
                return

            # 计算高斯核密度估计
            density = gaussian_kde(valid_values)
            density_values = density(valid_values)
            threshold = np.percentile(density_values, 5)  # 下5%密度阈值

            # 识别密度低于阈值的索引
            outlier_indices = valid_values.index[density_values < threshold]
            outlier_values = valid_values.loc[outlier_indices].unique()
            if len(outlier_values) > 0:
                logging.info(f"列 {column} 异常值检测到DMVs：{outlier_values}")
            self.dmv_results[column].update(outlier_values)
        except Exception as e:
            logging.error(f"列 {column} 的异常值检测失败: {e}")

    def detect_repeated_dmvs(self, column):
        """
        使用重复模式检测伪装缺失值
        当某个值在列中出现频率超过总行数10%时，认为该值可能为DMV
        :param column: 待检测的列名
        """
        values = self.data[column].astype(str)
        value_counts = Counter(values)
        total_rows = len(self.data)
        dmvs = {value for value, count in value_counts.items() if count > total_rows * 0.1}
        if dmvs:
            logging.info(f"列 {column} 重复检测到DMVs：{dmvs}")
        self.dmv_results[column].update(dmvs)

    def detect_random_missing_dmvs_fast(self, column):
        """
        使用Fast DiMaC方法检测随机缺失值 (MAR DMVs)
        仅对唯一值数大于2且不全唯一的列进行检测
        :param column: 待检测的列名
        """
        unique_count = self.data[column].nunique(dropna=True)
        if unique_count <= 2 or unique_count == len(self.data):
            logging.info(f"列 {column} 因唯一值数量过低（{unique_count}）跳过Fast DiMaC检测")
            return

        table_size = len(self.data)
        index = defaultdict(list)
        for idx, value in self.data[column].items():
            index[value].append(idx)

        scores = {}
        for value, indices in index.items():
            # 计算当前值的出现概率
            prob_value = len(indices) / table_size
            other_values = [v for v in index.keys() if v != value]
            # 计算其他值的联合概率（乘积近似）
            if other_values:
                prob_others = np.prod([len(index[v]) / table_size for v in other_values])
            else:
                prob_others = 1
            correlation = prob_value / prob_others if prob_others > 0 else 0
            # 简单计算评分，后续可根据论文公式进行进一步调整
            score = sum(
                (len(index[v]) / table_size) / (1 + abs(correlation - prob_value))
                for v in other_values
            )
            scores[value] = score

        if scores:
            threshold = np.percentile(list(scores.values()), 95)  # 取最高5%作为阈值
            dmvs = {value for value, score in scores.items() if score >= threshold}
            if dmvs:
                logging.info(f"列 {column} Fast DiMaC检测到DMVs：{dmvs}")
            self.dmv_results[column].update(dmvs)

    def run_detection(self):
        """
        运行所有模块检测伪装缺失值
        """
        for column in self.data.columns:
            self.detect_syntactical_dmvs(column)
            # 仅对数值类型的列进行异常值检测
            if pd.api.types.is_numeric_dtype(self.data[column]):
                self.detect_outlier_dmvskde(column)
            self.detect_repeated_dmvs(column)
            self.detect_random_missing_dmvs_fast(column)
        return self.dmv_results


# def clean_fields_to_replace(df):
#     """
#     对fields_to_replace字段进行清洗，确保格式正确
#     :param df: DataFrame
#     :return: 清洗后的DataFrame
#     """
#     # 如果字段为列表，则转为以逗号分隔的字符串
#     df['fields_to_replace'] = df['fields_to_replace'].apply(
#         lambda x: ', '.join(x) if isinstance(x, list) else x
#     )
#     return df


def FAHES_error_detection(dirty_data_path):
    """
    数据处理流程：加载数据、检测DMVs、添加错误标记、保存处理后的数据
    :param dirty_data_path: 输入文件路径
    :return: 处理后的DataFrame
    """
    # 加载数据
    df = pd.read_csv(dirty_data_path)
    output_path = 'FAHES_output.csv'

    # 初始化FAHES算法
    fahes = FAHES(df)
    dmv_results = fahes.run_detection()

    # 添加fields_to_replace和weight列
    def identify_errors(row):
        errors = []
        for col in df.columns:
            # 对比时统一去除首尾空格，保证字符串匹配
            if col in dmv_results and str(row[col]).strip() in {str(val).strip() for val in dmv_results[col]}:
                errors.append(col)
        return errors

    df['fields_to_replace'] = df.apply(identify_errors, axis=1)
    df['weight'] = df['fields_to_replace'].apply(lambda x: 1 if x else 0)
    # 格式化 fields_to_replace 字段为类似 [col1, col2] 的字符串
    df['fields_to_replace'] = df['fields_to_replace'].apply(
        lambda x: f"[{', '.join(x)}]" if x else "[]"
    )

    df = clean_fields_to_replace(df)

    # 保存处理后的数据
    df.to_csv(output_path, index=False)
    print(f"处理后的数据已保存到 {output_path}")
    return df

def KATARA_error_detection(clean_data_path, dirty_data_path, knowledge_base_path, dataset_name):
    """
    使用知识库进行错误检测，并将结果保存到文件。
    返回:
    - 包含权重列的 Pandas DataFrame
    """
    try:
        # 加载数据
        sf_clean = pd.read_csv(clean_data_path)
        sf_dirty = pd.read_csv(dirty_data_path)
        output_path = 'KATARA_output.csv'

        # 加载知识库
        with open(knowledge_base_path, "r") as f:
            knowledge_base = json.load(f)

        # 如果知识库是列表，将其包装成一个字典
        if isinstance(knowledge_base, list):
            knowledge_base = {
                "association_rules": knowledge_base,
                "functional_dependencies": [],
                "cfds": [],
                "dcs": [],
                "etl_rules": [],
                "mds": [],
                "contextual_rules": []
            }

        # 过滤适用于当前数据集的规则
        filtered_knowledge_base = {
            key: [rule for rule in rules if rule.get("dataset") == dataset_name]
            for key, rules in knowledge_base.items()
        }

        # 预处理数据类型
        numeric_columns = sf_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
        for col in numeric_columns:
            sf_dirty[col] = pd.to_numeric(sf_dirty[col], errors='coerce')

        def detect_and_replace_errors(row):
            """
            检测单行数据是否违反知识库中的规则
            """
            fields_to_replace = []
            #weight = 0

            # 7. 上下文规则检查
            for contextual_rule in filtered_knowledge_base.get("contextual_rules", []):
                try:
                    field = contextual_rule["field"]
                    value = contextual_rule["value"]
                    context_field = contextual_rule["context_field"]
                    expected_context_value = contextual_rule["context_value"]

                    field_value = str(row.get(field, ""))
                    context_value = str(row.get(context_field, ""))

                    if field_value == str(value) and context_value != str(expected_context_value):
                        fields_to_replace.append(context_field)
                        weight = 1
                except Exception as e:
                    print(f"单个上下文规则处理错误: {str(e)}")
                    continue

            weight = 1 if fields_to_replace else 0

            return list(set(fields_to_replace)), weight

        # 应用错误检测
        try:
            results = [detect_and_replace_errors(row) for _, row in sf_dirty.iterrows()]
            sf_dirty['fields_to_replace'], sf_dirty['weight'] = zip(*results)
        except Exception as e:
            print(f"错误检测应用失败: {str(e)}")
            return None

        # 保存结果
        try:
            sf_dirty.to_csv(output_path, index=False)
            print(f"错误检测完成，结果已保存到 {output_path}")
        except Exception as e:
            print(f"结果保存失败: {str(e)}")

        return sf_dirty

    except Exception as e:
        print(f"KATARA 错误检测失败: {str(e)}")
        return None


# 处理缺失值：数值型数据用均值填充，分类型数据用众数填充
def handle_missing_values(df):
    # 数值型数据用均值填充
    df_numeric = df.select_dtypes(include=[np.number])  # 选择数值型列
    df[df_numeric.columns] = df_numeric.fillna(df_numeric.mean())

    # 分类型数据用众数填充
    df_categorical = df.select_dtypes(exclude=[np.number])  # 选择分类型列
    for col in df_categorical.columns:
        df[col] = df[col].fillna(df[col].mode()[0])  # 众数填充

    return df

# 计算相似性矩阵
def compute_similarity(df):
    # 处理缺失值
    df = handle_missing_values(df)  # 填充缺失值
    # 对类别特征进行One-hot编码
    df_encoded = pd.get_dummies(df)
    # 数据归一化
    scaler = MinMaxScaler()
    df_normalized = scaler.fit_transform(df_encoded)

    # 计算余弦相似度
    similarity_matrix = cosine_similarity(df_normalized)
    return similarity_matrix


# 拟合高斯混合模型
def fit_gmm(similarity_matrix):
    # 确保数据集有至少两个样本
    if similarity_matrix.shape[0] < 2:
        raise ValueError("The similarity matrix must have at least two samples.")
    gmm = GaussianMixture(n_components=2)
    gmm.fit(similarity_matrix)
    return gmm


# 检测重复值并添加字段
def ZeroER_error_detection(df):
    df = pd.read_csv(df)
    output_path = 'ZeroER_output.csv'

    similarity_matrix = compute_similarity(df)
    gmm = fit_gmm(similarity_matrix)

    # 预测每对记录是匹配（重复）还是不匹配
    probabilities = gmm.predict_proba(similarity_matrix)[:, 0]

    # 创建新的列 fields_to_replace 和 weight
    fields_to_replace = []
    weight = []

    # for index, prob in enumerate(probabilities):
    #     if prob > 0.6:  # 高于0.5认为是重复值
    #         # 如果是重复行，将这一行的所有列名存储到 fields_to_replace 中
    #         #fields_to_replace.append(", ".join(df.columns))  # 存储所有列名，逗号分隔f'["{",".join(row_errors)}"]'
    #         fields_to_replace.append(f'["{",".join(df.columns)}"]')
    #         weight.append(1)
    #     else:
    #         fields_to_replace.append(None)
    #         weight.append(0)

    n_rows = df.shape[0]  # 获取数据行数

    # 用于记录哪些行已经被比较过
    compared_rows = set()

    for index in range(n_rows):
        if index in compared_rows:  # 如果当前行已经比较过，跳过
            continue

        # 对比当前行与之后的每一行
        for compare_index in range(index + 1, n_rows):
            if compare_index in compared_rows:  # 如果已经比较过该行，跳过
                continue

            prob = similarity_matrix[index, compare_index]  # 获取当前行与比较行的相似度概率

            if prob > 0.95:  # 如果认为是重复值
                # 将当前行和比较行的所有列名存储到 fields_to_replace
                fields_to_replace.append(", ".join(df.columns))  # 存储所有列名，逗号分隔
                weight.append(1)

                # 标记当前行和比较行已比较
                compared_rows.add(index)
                compared_rows.add(compare_index)

        if index not in compared_rows:  # 如果当前行没有和任何其他行匹配
            fields_to_replace.append(None)
            weight.append(0)

        # 确保 fields_to_replace 和 weight 列长度与数据框行数一致
        # 如果长度不一致，填充 None 或其他默认值直到长度匹配
    while len(fields_to_replace) < n_rows:
        fields_to_replace.append(None)
        weight.append(0)

    df["fields_to_replace"] = fields_to_replace
    df['weight'] = weight

    df = clean_fields_to_replace(df)

    df.to_csv(output_path, index=False)
    print(f"检测后的数据已保存到 {output_path}")

    return df



#df = pd.read_csv('nasa_60%.csv')
result1 = nadeef_error_detection(clean_data_path, dirty_data_path,dataset_name='retailSales')
result2 = dboost_error_detection(clean_data_path, dirty_data_path)
#result = outlier_error_detection(clean_data_path, dirty_data_path)
#result = mv_error_detection(dirty_data_path)
#result = RandomForest_error_detection(dirty_data_path,clean_data_path)
#result = FAHES_error_detection(dirty_data_path)
result3 = KATARA_error_detection(clean_data_path, dirty_data_path,knowledge_base_path,dataset_name='retailSales')
result4 = ZeroER_error_detection(dirty_data_path)
# if result is None:
#     print("Error: error detection returned None.")
# else:
#     print(f"detected data shape: {result.shape}")
