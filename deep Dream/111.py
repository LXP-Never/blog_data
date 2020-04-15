import numpy as np

k = np.float32([1, 4, 6, 4, 1])
k = np.outer(k, k)  # 计算两个向量的外积(5, 5)
k5x5 = k[:, :, None, None] / k.sum() * np.eye(3, dtype=np.float32)
print(k5x5.shape)
print(k5x5)     # (5, 5, 3, 3)
