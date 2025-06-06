"""
test custom code smell detection for missing randomness seed
"""


import random
import numpy as np
import tensorflow as tf

# Function using randomness but NOT setting seed
x = random.random()
y = np.random.rand()
z = tf.random.uniform((1,))
