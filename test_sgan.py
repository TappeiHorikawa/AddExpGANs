import numpy as np
import tensorflow as tf
import datetime
from keras import backend as K

iterations = 4000
batch_size = 256
sample_interval = 800

sgan = SGAN()
sgan.train(iterations, batch_size, sample_interval)
