import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
data = pd.read_csv('../dataset/DOF_dataset/DOF.csv')

# 添加一个新列用于分组
data['group'] = data['image'].str.extract('_(\d).jpg')[0]

# 设置图表的大小和分辨率
plt.figure(figsize=(15, 5), dpi=300)  # 调整图表大小和分辨率

# 定义每个组的子图位置
subplot_positions = {'0': 1, '1': 2, '2': 3}

# 对每个组绘制直方图
for group in ['0', '1', '2']:
    group_data = data[data['group'] == group]
    plt.subplot(1, 3, subplot_positions[group])  # 1行3列子图
    plt.hist(group_data['score'], bins=np.arange(0, 10.1, 0.1), range=(0, 10), edgecolor='black', alpha=0.8, color='darkblue')
    plt.title(f'Group {group} - Histogram of Average Scores')
    plt.xlabel('Average Scores')
    plt.ylabel('Frequency')

# 调整子图间距
plt.tight_layout()
# 保存图表到PNG文件
plt.savefig('average_scores_histogram_by_group.png')
# 显示图表
plt.show()
