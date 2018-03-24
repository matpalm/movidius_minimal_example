import numpy as np

np.random.seed(123)

# tensor to train for 1.0
POS_TENSOR = np.random.random(size=(64, 64, 3))

# tensor to train for 0.0
NEG_TENSOR = np.random.random(size=(64, 64, 3))
