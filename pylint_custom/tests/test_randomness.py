# import random
# import numpy as np
# from sklearn.model_selection import train_test_split, KFold

# # 1) missing random seed
# random.shuffle([1, 2, 3])
# x = np.random.rand(5)

# # 2) missing random_state
# train_test_split([[1, 2], [3, 4]], [0, 1])
# KFold()

# # 3) good practice
# random.seed(0)
# np.random.seed(0)
# train_test_split([[1, 2], [3, 4]], [0, 1], random_state=42)
# KFold(random_state=0)
