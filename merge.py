# import pandas as pd
# import ast
#
# # 读取两个CSV文件
# fahes_df = pd.read_csv(r'C:\Users\MY\PycharmProjects\v\merged_output.csv')
# dboost_df = pd.read_csv(r'C:\Users\MY\PycharmProjects\v\ZeroER_output.csv')
#
#
# def merge_fields(row1, row2):
#     try:
#         # 将字符串转换为Python列表
#         list1 = ast.literal_eval(str(row1)) if pd.notna(row1) and str(row1).strip() != '[]' else []
#         list2 = ast.literal_eval(str(row2)) if pd.notna(row2) and str(row2).strip() != '[]' else []
#
#         # 合并列表并去重，保持原有顺序
#         merged = list1 + [x for x in list2 if x not in list1]
#
#         # 返回字符串形式的列表
#         return str(merged)
#     except:
#         return '[]'
#
#
# # 创建结果DataFrame
# result_df = fahes_df.copy()
#
# # 逐行合并fields_to_replace列
# for index in result_df.index:
#     if index < len(dboost_df):
#         fahes_fields = fahes_df.loc[index, 'fields_to_replace']
#         dboost_fields = dboost_df.loc[index, 'fields_to_replace']
#         result_df.loc[index, 'fields_to_replace'] = merge_fields(fahes_fields, dboost_fields)
#
# # 保存合并后的结果到新的CSV文件
# output_path = r'C:\Users\MY\PycharmProjects\v\merged_output.csv'
# result_df.to_csv(output_path, index=False)

import pandas as pd
import ast

# 读取四个CSV文件
file_paths = [
    r'C:\Users\MY\PycharmProjects\v\dBoost_output.csv',
    r'C:\Users\MY\PycharmProjects\v\NADEEF_output.csv',
    # 请在下面添加另外两个CSV文件的路径
    r'C:\Users\MY\PycharmProjects\v\KATARA_output.csv',
    r'C:\Users\MY\PycharmProjects\v\ZeroER_output.csv'
]

# 读取所有CSV文件
dataframes = [pd.read_csv(path) for path in file_paths]

def merge_multiple_fields(rows):
    """
    合并多个字段的函数
    rows: 包含多个fields_to_replace值的列表
    """
    merged = []
    try:
        for row in rows:
            if pd.notna(row) and str(row).strip() != '[]':
                # 将字符串转换为Python列表
                current_list = ast.literal_eval(str(row))
                # 添加新的元素（避免重复）
                merged.extend(x for x in current_list if x not in merged)
        return str(merged)
    except:
        return '[]'

# 创建结果DataFrame，使用第一个文件作为基础
result_df = dataframes[0].copy()

# 获取所有DataFrame中的最小行数
min_rows = min(len(df) for df in dataframes)

# 逐行合并fields_to_replace列
for index in range(min_rows):
    # 获取所有文件中当前行的fields_to_replace值
    fields_to_merge = [df.loc[index, 'fields_to_replace'] for df in dataframes]
    # 合并这些值
    result_df.loc[index, 'fields_to_replace'] = merge_multiple_fields(fields_to_merge)

# 保存合并后的结果到新的CSV文件
output_path = r'C:\Users\MY\PycharmProjects\v\merged_output.csv'
result_df.to_csv(output_path, index=False)