import numpy as np
from scipy.sparse import csr_matrix

# Tạo một mảng numpy
arr = np.array([[0, 0, 0, 0, 0, 1, 1, 0, 2], [0, 2, 5, 0, 6, 0, 0, 4, 0]])

print(arr)

# Chuyển đổi mảng numpy thành ma trận thưa CSR
sparse_arr = csr_matrix(arr)

# In ra ma trận thưa CSR
print(sparse_arr)
