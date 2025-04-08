import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_process import extract_accelerate_rate

# 设置数据示例
np.random.seed(42)
# speedup_translated = np.random.normal(loc=1.2, scale=0.5, size=1000)
# speedup_optimized = np.random.normal(loc=2.0, scale=0.8, size=1000)
speedup_translated = extract_accelerate_rate('/code/LLM4HPCTransCompile/Results/QwenCoder_14b/3_eval_results.json')
speedup_optimized = extract_accelerate_rate('/code/LLM4HPCTransCompile/Results/DeepSeekCoder_Lite/3_eval_results.json')
print(speedup_translated)
# 绘制直方图
plt.figure(figsize=(8, 5))
plt.hist(speedup_translated, bins=50, alpha=0.5, density=True, color='blue', label='Translated')
plt.hist(speedup_optimized, bins=50, alpha=0.5, density=True, color='red', label='Optimized')

# 叠加拟合曲线
sns.kdeplot(speedup_translated, color='blue', linewidth=2, label='Translated')
sns.kdeplot(speedup_optimized, color='red', linewidth=2, label='Optimized')

# 添加参考线
plt.axvline(x=1.0, color='black', linestyle='--', label='PyTorch Native')

# 图例和标签
plt.title("PyTorch Native Speedup - Level 3 Kernels", fontsize=14)
plt.xlabel("Speedup (x faster than PyTorch native)", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.xlim(0, 4)

# 显示图像
plt.tight_layout()
plt.savefig('./pictures/speed_up_3.pdf', dpi=300)