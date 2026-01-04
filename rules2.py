# 自动提取关联规则，函数依赖，条件函数依赖，否认约束，ETL规则，匹配依赖，上下文信息
import pandas as pd
import numpy as np
import mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import json


# 读取数据文件
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("数据加载成功！")
        return data
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None


def mine_association_rules(data, min_support=0.1, min_confidence=0.6):
    # 分别处理数值型和分类型数据
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    # 处理数值型数据：将数值型数据离散化
    bool_data = data.copy()
    for col in numeric_cols:
        # 使用四分位数进行离散化
        q75, q25 = bool_data[col].quantile(0.75), bool_data[col].quantile(0.25)
        bool_data[f"{col}_high"] = bool_data[col] > q75
        bool_data[f"{col}_low"] = bool_data[col] < q25
        bool_data = bool_data.drop(columns=[col])

    # 处理分类型数据：使用one-hot编码
    bool_data = pd.get_dummies(bool_data, columns=categorical_cols)

    # 转换为布尔型
    bool_data = bool_data.astype(int)

    # 计算频繁项集
    frequent_itemsets = apriori(bool_data, min_support=min_support, use_colnames=True)

    # 检查频繁项集是否生成成功
    if frequent_itemsets.empty:
        print("未发现任何频繁项集，请调整 min_support 参数")
        return None

    # 打印频繁项集
    # print("频繁项集：")
    # print(frequent_itemsets)

    # 生成关联规则
    try:
        num_itemsets = len(frequent_itemsets)
        rules = association_rules(frequent_itemsets, num_itemsets=num_itemsets, metric="confidence",
                                  min_threshold=min_confidence)

        # 过滤规则，增加支持度和置信度的阈值
        rules = rules[(rules['support'] >= 0.1) & (rules['confidence'] >= 0.6)]

        print("关联规则生成成功！")
        return rules
    except Exception as e:
        print(f"关联规则生成失败: {e}")
        return None


def detect_functional_dependencies(data, threshold=0.3):
    dependencies = []
    columns = data.columns.tolist()

    for i, col1 in enumerate(columns):
        for col2 in columns:
            if col1 != col2:
                try:
                    # 计算互信息
                    if data[col1].dtype in ['int64', 'float64'] and data[col2].dtype in ['int64', 'float64']:
                        data[col1] = pd.to_numeric(data[col1], errors='coerce')
                        data[col2] = pd.to_numeric(data[col2], errors='coerce')
                        mutual_info = mutual_info_regression(data[[col1]], data[col2])
                    else:
                        mutual_info = mutual_info_classif(data[[col1]].astype(str), data[col2].astype(str))

                    if mutual_info[0] > threshold:  # 设定依赖的阈值
                        dependencies.append((col1, col2, mutual_info[0]))
                except Exception as e:
                    print(f"处理列 {col1} 和 {col2} 时出错: {e}")
                    continue

    print("函数依赖检测完成！")
    return dependencies


def extract_cfds(data):
    cfds = []

    for condition_col in data.columns:
        for dependent_col in data.columns:
            if condition_col != dependent_col:
                try:
                    # 对数值型列进行离散化处理
                    if data[condition_col].dtype in ['int64', 'float64']:
                        data[condition_col] = pd.to_numeric(data[condition_col], errors='coerce')
                        q75, q25 = data[condition_col].quantile(0.75), data[condition_col].quantile(0.25)
                        conditions = [
                            (data[condition_col] <= q25, "low"),
                            (data[condition_col] > q75, "high"),
                            ((data[condition_col] > q25) & (data[condition_col] <= q75), "medium")
                        ]

                        for condition, label in conditions:
                            if condition.any():
                                subset = data[condition][dependent_col]
                                if len(subset.unique()) == 1:
                                    cfds.append({
                                        "condition": f"{condition_col} is {label}",
                                        "dependent": dependent_col,
                                        "determined_by": condition_col,
                                        "value": str(subset.iloc[0])
                                    })
                    else:
                        # 分类型数据直接分组
                        grouped = data.groupby(condition_col)[dependent_col].unique()
                        for condition, values in grouped.items():
                            if len(values) == 1:
                                cfds.append({
                                    "condition": f"{condition_col} = {condition}",
                                    "dependent": dependent_col,
                                    "determined_by": condition_col,
                                    "value": str(values[0])
                                })
                except Exception as e:
                    print(f"处理列 {condition_col} 和 {dependent_col} 时出错: {e}")
                    continue

    print("条件函数依赖提取完成！")
    return cfds

# 自定义序列化函数，处理 frozenset 类型
def frozenset_to_list(obj):
    if isinstance(obj, frozenset):
        return list(obj)  # 将 frozenset 转换为 list
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# 构建知识库
def build_knowledge_base(association_rules, functional_dependencies, cfds, dataset_name):
    # 转换关联规则为可序列化的形式
    association_rules_serializable = association_rules.map(
        lambda x: list(x) if isinstance(x, frozenset) else x
    ).to_dict(orient="records")

    # 为关联规则添加 dataset 字段
    for rule in association_rules_serializable:
        rule["dataset"] = dataset_name

    # 转换函数依赖为可序列化的形式
    functional_dependencies_serializable = [
        {
            "dependent": dep[0],
            "determined_by": dep[1],
            "mutual_info": dep[2],
            "dataset": dataset_name
        }
        for dep in functional_dependencies
    ]

    # 转换条件函数依赖（CFDs）为可序列化的形式
    cfds_serializable = [
        {**cfd, "dataset": dataset_name} for cfd in cfds
    ]


    # 组装知识库
    knowledge_base = {
        "association_rules": association_rules_serializable,
        "functional_dependencies": functional_dependencies_serializable,
        "cfds": cfds_serializable
    }

    # 自定义序列化函数，处理 int64 类型
    def serialize_int64(obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return obj

    # 替换 DataFrame 中的 Infinity 为一个大数
    # def replace_infinity_with_large_number(df):
    #     return df.map(lambda x: 1e9 if x == np.inf else x)
    def replace_infinity_with_large_number(obj):
        if isinstance(obj, (int, float)) and obj == float("inf"):
            return 1e9
        elif isinstance(obj, dict):
            return {k: replace_infinity_with_large_number(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_infinity_with_large_number(x) for x in obj]
        elif isinstance(obj, np.int64):  # 处理 np.int64 类型
            return int(obj)
        return obj

    knowledge_base = replace_infinity_with_large_number(knowledge_base)

    # 保存为 JSON 文件
    with open("knowledge_base.json", "w") as f:
        json.dump(knowledge_base, f, indent=4, default=serialize_int64)

    print("知识库构建完成！")
    return knowledge_base


# 整合多个数据集的知识库
def build_global_knowledge_base(datasets):
    global_knowledge_base = {
        "association_rules": [],
        "functional_dependencies": [],
        "cfds": [],
    }

    for dataset_name, data in datasets.items():
        # 挖掘关联规则
        rules = mine_association_rules(data)
        # 检测函数依赖
        dependencies = detect_functional_dependencies(data)
        # 提取条件函数依赖（CFDs）
        cfds = extract_cfds(data)


        dataset_knowledge_base = build_knowledge_base(
            rules, dependencies, cfds, dataset_name
        )

        # 将单个知识库合并到全局知识库中
        for key in global_knowledge_base.keys():
            global_knowledge_base[key].extend(dataset_knowledge_base[key])

    # 保存全局知识库为 JSON 文件
    with open("global_knowledge_base.json", "w") as f:
        json.dump(global_knowledge_base, f, indent=4)

    print("全局知识库构建完成！")
    return global_knowledge_base


# 主程序
datasets = {
    "smartfactory": load_data("sf_clean.csv"),
    "nasa": load_data("nasa_clean.csv"),
    "flight": load_data("flight_clean.csv"),
    "retailSales": load_data("retailSales_clean.csv"),
    "hospital": load_data("hospital_clean.csv"),
    "wdbc": load_data("wdbc_clean.csv"),
    "beer": load_data("beers_clean.csv")
}

# 过滤掉加载失败的数据集
datasets = {name: data for name, data in datasets.items() if data is not None}
# 构建知识库
knowledge_base = build_global_knowledge_base(datasets)
# print(knowledge_base)
# print(json.dumps(knowledge_base, indent=4))