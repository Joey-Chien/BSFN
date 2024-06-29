import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('../dataset/DOF_dataset/DOF.csv')

# 初始化差值計數器
diff_counts = defaultdict(int)

# 初始化兩個列表來存儲差值
diff_0_1_list = []
diff_0_2_list = []

# 分組遍歷
for i in range(0, len(df), 3):  # 假設每組的資料都是連續的且完整
    score_0 = df.iloc[i]["score"]
    score_1 = df.iloc[i+1]["score"]
    score_2 = df.iloc[i+2]["score"]


    # # 計算差值
    # diff_0_1 = score_0 - score_1
    # diff_0_2 = score_0 - score_2

    # 計算差值並四捨五入
    diff_0_1 = score_0 - score_1
    diff_0_2 = score_0 - score_2

    if diff_0_1 > 10 or diff_0_1 < -10:
        print('Error', i, 1)
    if diff_0_2 > 10 or diff_0_2 < -10:
        print('Error', i , 2)




    # 加入列表
    diff_0_1_list.append(diff_0_1)
    diff_0_2_list.append(diff_0_2)

    # 計算平均差值
    mean_diff_0_1 = sum(diff_0_1_list) / len(diff_0_1_list) if diff_0_1_list else 0
    mean_diff_0_2 = sum(diff_0_2_list) / len(diff_0_2_list) if diff_0_2_list else 0

    # 更新計數器
    diff_counts[diff_0_1] += 1
    diff_counts[diff_0_2] += 1


print('len: ', len(diff_0_1_list))

temp_diff_0_1 = np.array(diff_0_1_list)
print(temp_diff_0_1)

print(f"Average of diff_0_1: {mean_diff_0_1}")
print(f"Average of diff_0_2: {mean_diff_0_2}")

print('std: ', np.std(temp_diff_0_1))
# 輸出結果
# print(dict(diff_counts))

differences = list(diff_counts.keys())
counts = [diff_counts[d] for d in differences]

# 繪製長條圖
plt.figure(figsize=(10, 6))
plt.bar(differences, counts, width=0.02, color='darkblue')  # 寬度設為0.01以更好地顯示差異
plt.xlabel('Difference in Average Scores')
plt.ylabel('Count')
plt.title('Histogram of Score Differences')
plt.grid(True)
plt.show()

# 调整子图间距
plt.tight_layout()
# 保存图表到PNG文件
plt.savefig('minus.png')
# 显示图表
plt.show()
