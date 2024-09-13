import pandas as pd
import numpy as np
import tensorflow as tf

# Display CUDA version
print('CUDA version:', tf.test.is_built_with_cuda())

# Display cuDNN version
print('cuDNN version:', tf.test.is_built_with_cudnn())
