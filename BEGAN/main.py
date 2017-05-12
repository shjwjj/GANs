import tensorflow as tf
import numpy as np

import os
from celeba_began import BEgan

def main(_):
	with tf.Session() as sess: 
		began = BEgan(sess = sess)

		# began.train()

		began.test()

if __name__ == '__main__':
	tf.app.run()