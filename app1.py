import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

import tensorflow as tf
print("Devices available: ", tf.config.list_physical_devices('GPU'))
