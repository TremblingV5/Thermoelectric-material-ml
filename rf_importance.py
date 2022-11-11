import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

df_src = pd.read_csv("./ucsb_test.csv")

# 删除一些不是很有用的列
for key in ["Unnamed: 0", "formula", "crystallinity", "composition", "synthesis"]:
    del df_src[key]

zt = df_src["zT"]
del df_src["zT"]

zt = zt.to_numpy().astype("int")
df = df_src.to_numpy().astype("int")

# 定义随机森林分类器并进行拟合
rf = RandomForestClassifier()
rf.fit(df, zt)

# 变量名
feature_names = df_src.columns
# 重要性
importances = rf.feature_importances_
# 标准差
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

# 重要性和变量名一一对应并保存
forest_importances = pd.Series(importances, index=feature_names)
forest_importances.to_csv("importances.csv")

# 以下是画图的一些代码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(24, 8))
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("基于MDI的变量重要程度计算")
ax.set_ylabel("纯度下降均值")
fig.tight_layout()

plt.savefig("result.jpg")
