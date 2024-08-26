import matplotlib.pyplot as plt
import numpy as np

# Sample data
np.random.seed(0)
indices = np.arange(0, 650)
similarity_scores = np.random.rand(650)

# Classify the data into non-outliers, outliers, and middle values
non_outliers = similarity_scores > 0.75
outliers = similarity_scores < 0.25
middle_values = (similarity_scores >= 0.25) & (similarity_scores <= 0.75)

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(indices[non_outliers], similarity_scores[non_outliers], color='blue', label='Non-outliers', marker='x')
plt.scatter(indices[outliers], similarity_scores[outliers], color='red', label='Outliers', marker='x')
plt.scatter(indices[middle_values], similarity_scores[middle_values], color='green', label='Middle values', marker='x')

# Add labels and title
plt.xlabel('Index')
plt.ylabel('Similarity Score')
plt.title('Scatter Plot of similarity test with Outliers and Middle Values')
plt.legend()
plt.grid(True)

# Save the figure as a .png file
output_path = '/code/LLM4HPCTransCompile/plot/performance.png'
plt.savefig(output_path)

plt.show()
