import pandas as pd
import ast

# 读取 CSV 文件
df = pd.read_csv("outlier_output.csv")

# 假设目标列名为 "col"
def remove_duplicates(cell):
    try:
        # 把字符串转为列表
        lst = ast.literal_eval(cell)
        if isinstance(lst, list):
            # 去重并保持顺序
            lst = list(dict.fromkeys(lst))
            return str(lst)
    except:
        pass
    return cell

# 对目标列应用去重函数
df["fields_to_replace"] = df["fields_to_replace"].apply(remove_duplicates)

# 保存到新 CSV 文件
df.to_csv("outlier_output.csv", index=False)



# 读取 Excel
df = pd.read_csv('ZeroER_output.csv')

# 解析 G 列的字符串为 Python 列表，并统计长度
df["count"] = df["fields_to_replace"].apply(lambda x: len(ast.literal_eval(x)) if pd.notna(x) else 0)

# 计算总和
total = df["count"].sum()

print("总和:", total)