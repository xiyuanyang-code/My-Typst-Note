import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. 训练数据集
# A, B, C 是特征，Target 是目标变量
data = {
    "A": [1, 0, 0, 0, 1, 1, 1, 1],
    "B": [2, 1, 0, 2, 1, 0, 2, 1],
    "C": [0, 0, 1, 0, 0, 1, 1, 0],
    "Target": [1, 0, 0, 1, 1, 0, 0, 1],
}

df_train = pd.DataFrame(data)
X_train = df_train[["A", "B", "C"]]
y_train = df_train["Target"]


# 2. 创建和训练决策树模型
# 使用相同的 random_state 确保与之前训练的模型一致
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

test_data = {"A": [1, 1, 0, 0, 0, 1], "B": [1, 2, 1, 2, 1, 1], "C": [0, 0, 0, 0, 1, 1]}

df_test = pd.DataFrame(test_data)
pred_ans = clf.predict(df_test)
print(pred_ans)

# # 3. 决策树可视化
# # 设置较大的图表尺寸，以便清晰地看到所有节点和文本
# plt.figure(figsize=(15, 10))

# # 使用 plot_tree 函数绘制决策树
# plot_tree(
#     clf,
#     feature_names=feature_names,
#     class_names=class_names,
#     filled=True,  # 用颜色填充节点，更直观
#     rounded=True, # 节点使用圆角
#     fontsize=10
# )

# plt.title("Decision Tree Structure for A, B, C Features", fontsize=14)
# # 显示图表
# plt.show()

# # 打印模型结构 (文本形式)
# print("\n--- 决策树结构 (文本形式) ---")
# from sklearn.tree import export_text
# tree_rules = export_text(clf, feature_names=feature_names)
# print(tree_rules)
