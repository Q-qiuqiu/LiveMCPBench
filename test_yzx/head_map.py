import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
# 读取 Excel
df = pd.read_excel("accuracy_table.xlsx", index_col=0)

# 绘制热力图
plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu")  # annot=True 显示数值
plt.title("Accuracy Heatmap")
plt.xlabel("insert_number")
plt.ylabel("TOP_TOOLS")
plt.tight_layout()
filename = f"heatmap.png"
save_dir="./test_yzx/"
save_path = os.path.join(save_dir, filename)
plt.savefig(save_path, dpi=300, bbox_inches="tight")  # 保存图片
print(f"图像已保存到: {save_path}")
