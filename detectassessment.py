import ast
import pandas as pd

dirty_data = pd.read_csv('best_dirty_data_episode_1.csv')
clean_data = pd.read_csv('retailSales_clean.csv')

correct_detection = 0
false_detection = 0  # 错误检测到的错误
missed_detection = 0  # 实际错误未检测到的错误
total_detected = 0

# 遍历每行数据，检查每个错误字段
for index, row in dirty_data.iterrows():
    detected_fields = ast.literal_eval(row['fields_to_replace'])  # 获取此行检测到的错误字段
    #detected_fields = row['fields_to_replace']  # 获取此行检测到的错误字段

    # 如果检测到错误，遍历所有错误字段
    if detected_fields != []:
        total_detected += len(detected_fields)

        # 检查每个字段是否为实际错误
        for field in detected_fields:
            # 跳过空字段名
            if field == '' or field not in dirty_data.columns:
                #print(f"跳过无效或空字段：{field}")  # 输出调试信息
                continue
            if row[field] != clean_data.at[index, field]:  # 如果脏数据与干净数据不同，则为错误
                correct_detection += 1
            else:
                false_detection += 1  # 误报：错误字段被误认为错误
    else:
        # 如果没有检测到错误，检查是否有实际错误
        for field in clean_data.columns:
            if pd.isnull(dirty_data.at[index, field]) and not pd.isnull(clean_data.at[index, field]):
                missed_detection += 1  # 漏检：实际错误未被检测到

# 计算准确率：正确检测的错误 / 总检测到的错误
precision = correct_detection / (correct_detection + false_detection) if (correct_detection + false_detection) > 0 else 0
recall = correct_detection / (correct_detection + missed_detection) if (correct_detection + missed_detection) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1 Score: {f1_score:.3f}')
