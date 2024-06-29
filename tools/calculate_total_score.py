import pandas as pd

# 讀取 CSV 檔案
df = pd.read_csv('../dataset/DOF_dataset/DOF.csv')

# 過濾出以 '_0.jpg' 結尾的資料並加總 'score'
sum_0 = df[df['image'].str.endswith('_0.jpg')]['score'].sum()

# 過濾出以 '_1.jpg' 結尾的資料並加總 'score'
sum_1 = df[df['image'].str.endswith('_1.jpg')]['score'].sum()

# 過濾出以 '_2.jpg' 結尾的資料並加總 'score'
sum_2 = df[df['image'].str.endswith('_2.jpg')]['score'].sum()

print(f"Sum of average scores for images ending with '_0.jpg': {sum_0}")
print(f"Sum of average scores for images ending with '_1.jpg': {sum_1}")
print(f"Sum of average scores for images ending with '_2.jpg': {sum_2}")

print(f"Sum of average scores for images ending with '_0.jpg': {sum_0/4400}")
print(f"Sum of average scores for images ending with '_1.jpg': {sum_1/4400}")
print(f"Sum of average scores for images ending with '_2.jpg': {sum_2/4400}")