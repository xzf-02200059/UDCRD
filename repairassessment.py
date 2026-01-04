import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score

# 计算评估指标
# 假设部分特征是分类特征，需要指定这些特征
clean_data = pd.read_csv('retailSales_clean.csv')

categorical_features = clean_data.select_dtypes(include=['object']).columns.tolist()

print(categorical_features)
numerical_features = [col for col in clean_data.columns if col not in categorical_features]
print(numerical_features)

# 读取修复后的数据和干净数据
repaired_data = pd.read_csv('student_best_dirty_data_iteration_4.csv')
repaired_data = repaired_data.drop(columns=['reconstruction_error'], errors='ignore')

# 初始化字典存储指标
metrics = {
    'Precision': {},
    'Recall': {},
    'F1-Score': {}
}

# 初始化全局 y_true 和 y_pred
global_y_true = []
global_y_pred = []

# 计算分类指标
for feature in numerical_features:
    if feature not in repaired_data.columns or feature not in clean_data.columns:
        print(f"Feature '{feature}' not found in data.")
        continue

    y_true = clean_data[feature]
    y_pred = repaired_data[feature]

    # y_true = y_true.dropna()
    # y_pred = y_pred.loc[y_true.index]  # 确保y_pred与y_true对应

    # 删除缺失值
    valid_mask = y_true.notna() & y_pred.notna()  # 创建一个布尔掩码，找到没有缺失值的索引
    y_true = y_true[valid_mask]  # 根据掩码筛选有效的 y_true
    y_pred = y_pred[valid_mask]  # 根据掩码筛选有效的 y_pred

    # 检查数据类型并进行适当转换
    if y_true.dtype.kind in 'if':  # 如果是整数或浮点类型
        # 对于数值型特征，我们需要将其转换为分类标签
        # 这里假设我们将数值四舍五入到最接近的整数作为类别
        y_true = y_true.round().astype(int)
        y_pred = y_pred.round().astype(int)
    elif y_true.dtype.kind == 'O':  # 如果是对象/字符串类型
        y_pred = y_pred.astype(str)
    else:
        print(f"Feature '{feature}' has unsupported dtype: {y_true.dtype}")
        continue

    # 计算精确率、召回率和 F1 分数
    try:
        # 确保数据不为空且有多个类别
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        if len(unique_classes) <= 1:
            print(f"Warning: Feature '{feature}' has only one class, skipping metrics calculation.")
            continue

        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        metrics['Precision'][feature] = precision
        metrics['Recall'][feature] = recall
        metrics['F1-Score'][feature] = f1

        # 将每个特征的值和特征名称一起存储，以便后续分析
        feature_ids = [feature] * len(y_true)
        global_y_true.extend(list(zip(feature_ids, y_true)))
        global_y_pred.extend(list(zip(feature_ids, y_pred)))
    except ValueError as e:
        print(f"Error processing feature '{feature}': {e}")


# 计算分类指标（包括类别型特征）
for feature in categorical_features:
    if feature not in repaired_data.columns or feature not in clean_data.columns:
        print(f"Feature '{feature}' not found in data.")
        continue

    y_true = clean_data[feature]
    y_pred = repaired_data[feature]

    # 转换为字符串类型（对于类别型数据）
    y_true = y_true.astype(str)
    y_pred = y_pred.astype(str)

    # 计算精确率、召回率和 F1 分数
    try:
        # 确保数据不为空且有多个类别
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        if len(unique_classes) <= 1:
            print(f"Warning: Feature '{feature}' has only one class, skipping metrics calculation.")
            continue

        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        metrics['Precision'][feature] = precision
        metrics['Recall'][feature] = recall
        metrics['F1-Score'][feature] = f1

        # 将每个特征的值和特征名称一起存储，以便后续分析
        feature_ids = [feature] * len(y_true)
        global_y_true.extend(list(zip(feature_ids, y_true)))
        global_y_pred.extend(list(zip(feature_ids, y_pred)))
    except ValueError as e:
        print(f"Error processing feature '{feature}': {e}")
# 计算修复前后的方差
# 计算每个数值特征的方差
feature_variances_before = {}
feature_variances_after = {}

for feature in numerical_features:
    if feature in clean_data.columns and feature in repaired_data.columns:
        # 计算修复前的方差
        values_before = clean_data[feature].dropna()
        if len(values_before) > 0:
            feature_variances_before[feature] = np.var(values_before)

        # 计算修复后的方差
        values_after = repaired_data[feature].dropna()
        if len(values_after) > 0:
            feature_variances_after[feature] = np.var(values_after)

# 计算全局方差
all_values_after = repaired_data[numerical_features].values.flatten()  # 将数值特征拉平成一维数组
all_values_before = clean_data[numerical_features].values.flatten()  # 将数值特征拉平成一维数组

# 去除 NaN 值
all_values_after = all_values_after[~np.isnan(all_values_after)]
all_values_before = all_values_before[~np.isnan(all_values_before)]

# 计算全局方差
global_variance_after = np.var(all_values_after)
global_variance_before = np.var(all_values_before)

# 输出全局方差
print(f"\n方差分析:")
print(f"修复后全局方差: {global_variance_after:.4f}")
print(f"修复前全局方差: {global_variance_before:.4f}")
print(f"修复前后全局方差比: {global_variance_before / global_variance_after:.4f}")

# 输出每个特征的方差
print("\n各特征方差分析:")
for feature in numerical_features:
    if feature in feature_variances_before and feature in feature_variances_after:
        variance_before = feature_variances_before[feature]
        variance_after = feature_variances_after[feature]
        variance_ratio = variance_before / variance_after if variance_after != 0 else float('inf')
        print(f"  {feature}: 修复前 {variance_before:.4f}, 修复后 {variance_after:.4f}, 比值 {variance_ratio:.4f}")

# 重新计算全局评估指标
# 将全局预测和真实值分开
features, y_true_values = zip(*global_y_true)
_, y_pred_values = zip(*global_y_pred)

# 转换为 NumPy 数组
global_y_true_values = np.array(y_true_values)
global_y_pred_values = np.array(y_pred_values)

# 计算全局评估指标
try:
    global_precision = precision_score(global_y_true_values, global_y_pred_values, average='weighted', zero_division=0)
    global_recall = recall_score(global_y_true_values, global_y_pred_values, average='weighted', zero_division=0)
    global_f1 = f1_score(global_y_true_values, global_y_pred_values, average='weighted', zero_division=0)

    # 输出全局评估指标
    print("\nGlobal Metrics:")
    print(f"Precision: {global_precision:.4f}")
    print(f"Recall: {global_recall:.4f}")
    print(f"F1-Score: {global_f1:.4f}")
except ValueError as e:
    print(f"Error calculating global metrics: {e}")
    print("Try checking if all features have the same number of classes and compatible data types.")

# 输出各特征的分类指标
print("\nClassification Metrics:")
for metric_name, metric_values in metrics.items():
    print(f"\n{metric_name}:")
    for feature, value in metric_values.items():
        print(f"  {feature}: {value:.4f}")