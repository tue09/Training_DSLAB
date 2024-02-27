import numpy as np

# Tạo một mảng 1D
a = np.array([1, 2, 3, 4, 5, 6])

# Sử dụng reshape(1, -1)
b = a.reshape(2, -1)

print(b)
# Kết quả: [[1 2 3 4 5 6]]
