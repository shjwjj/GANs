import time
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl
import math


from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/", one_hot=True)


batch_size = 16
z_dim = 128

def lrelu(x, leak = 0.3, name = 'lrelu'):
	f1 = 0.5*(1+leak)
	f2 = 0.5*(1-leak)
	return f1*x+f2*abs(x)


# def Generator(L_img, z):
# # L_img = tf.reshape(L_img,[batch_size,64, 64, 1])
# 	L = tcl.flattened(L_img)
# 	comb = tf.concat([L,z],1)
# 	with tf.variable_scope('gen') as scope:
# 		pre_conv1 = tcl.fully_connected(comb, 512, activation_fn = lrelu)
# 		pre_conv2 = tcl.fully_connected(pre_conv1, 256, activation_fn = lrelu)
# 		pre_conv3 = tcl.fully_connected(pre_conv2, 64, activation_fn = lrelu)
# 		pre_conv4 = tcl.fully_connected(pre_conv3, 32, activation_fn = lrelu)
# 		pre_conv5 = tcl.fully_connected(pre_conv4, 4, activation_fn = lrelu)
# 		de_conv = tcl.conv2d_transpose()
# 		deconv1 = tcl.conv2d_transpose(conv2, 128, kernel_size = 3, stride = 2,
# 			activation_fn=lrelu, normalizer_fn=tcl.batch_norm, 
# 	        padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
# 		images = (conv5+ 1.0)/2.0
# 	return images

def Generator(z):
	with tf.variable_scope('gen') as scope:
	    train = tcl.fully_connected(
	        z, 7 * 7 * 128, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
	    train = tf.reshape(train, (-1, 7, 7, 128))
	    train = tcl.conv2d_transpose(train, 64, 3, stride=2,
	                                activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
	    train = tcl.conv2d_transpose(train, 32, 3, stride=2,
	                                activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
	    train = tcl.conv2d_transpose(train, 1, 3, stride=1,
	                                activation_fn=tf.nn.tanh, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
	    train = (train+1.0)/2.0 * 255.0
	    return train

def AE(H_img,reuse = False):
	
	# L_img = tf.reshape(L_img, [batch_size, 64, 64, 3])
	with tf.variable_scope('crit') as scope:
		if reuse:
			scope.reuse_variables()
		conv1 = tcl.conv2d(H_img, num_outputs = 32, stride = 2, kernel_size = 3,		#28->14 32
			activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding = 'SAME')

		conv2 = tcl.conv2d(conv1, num_outputs = 64, stride = 2, kernel_size = 3,		#14->7	64
			activation_fn = tf.nn.relu, normalizer_fn = tcl.batch_norm, padding = 'SAME')

		pre = tcl.flatten(conv2)

		z = tcl.fully_connected(pre, num_outputs = 64, activation_fn = tf.nn.relu)

		post = tcl.fully_connected(z, num_outputs = 7*7*64, activation_fn = tf.nn.relu)

		post = tf.reshape(post, [batch_size, 7, 7, 64])							#4 256

		deconv1 = tcl.conv2d_transpose(post, 32, kernel_size = 3, stride = 2, 		#7->14  64->32
			activation_fn=lrelu, normalizer_fn=tcl.batch_norm, 
	        padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))

		deconv2 = tcl.conv2d_transpose(deconv1, 1, kernel_size = 3, stride = 2, 		#14->28  1
			activation_fn=tf.nn.tanh,
	        padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))

	return (deconv2+1.0)/2.0 * 255.0

 
tz = np.ones([20,128])
for i in range(20):
	tz[i] = tz[i]*(i)/20.0


class BEgan(object):

	def __init__(self, sess,
		image_size = 28,
		label_size = 28,
		batch_size = batch_size,
		channels = 1,
		# checkpoint_dir = '/media/shj/0BEE066D0BEE066D/DataSets/hevc/JEPGGAN/h5pyfile/q30train.h5'
		):

		self.sess = sess
		self.image_size = image_size
		self.label_size = label_size
		self.batch_size = batch_size
		self.channels = channels
		# self.checkpoint_dir = checkpoint_dir
		self.weight_mse = 10.0
		self.weight_GAN = 1.0
		self.kt = tf.Variable(0.0, trainable = False)
		self.kt_ = tf.minimum(tf.maximum(self.kt,0), 1.0)
		self.build_model()

	def compute_loss(self,ae_fake, ae_real, fake, real, gamma = 0.75):
		l2_fake_loss = tf.reduce_mean(tf.square(ae_fake - fake))
		l2_real_loss = tf.reduce_mean(tf.square(ae_real - real))
		D_loss = l2_real_loss - self.kt_ * l2_fake_loss
		G_loss = l2_fake_loss
		k_next = self.kt_ + 0.001 * (gamma * l2_real_loss - l2_fake_loss)
		measure = l2_real_loss + tf.abs(gamma * l2_real_loss - l2_fake_loss)
		return D_loss, G_loss, k_next, measure


	def build_model(self):
		# self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.channels], name = 'images')
		self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.channels], name = 'labels')
		self.z = tf.placeholder(tf.float32, [None, z_dim], name = 'z')
		self.fake = Generator(self.z)
		self.AE_ = AE(self.fake)
		self.AE = AE(self.labels, True)
		self.label0 = self.labels[0]
		self.fake0 = self.fake[0]
		self.c_0 = self.AE_[0]
		self.c0 = self.AE[0]

		self.D_loss, self.G_loss, self.k_next, self.measure = self.compute_loss(self.AE_, self.AE, self.fake, self.labels)
		
		
		self.kt_change = tf.assign(self.kt, self.k_next)
		
		tf.summary.scalar('kt', self.kt)
		tf.summary.scalar('kt_', self.kt_)

		tf.summary.scalar('D_loss', self.D_loss)
		tf.summary.scalar('G_loss', self.G_loss)
		tf.summary.scalar('measure', self.measure)

		self.saver = tf.train.Saver()

		self.theta_g = tf.get_collection(
    	tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen')
		self.theta_c = tf.get_collection(
    	tf.GraphKeys.TRAINABLE_VARIABLES, scope='crit')
		
	def next_feed_dict(self):
			batch = data.train.next_batch(self.batch_size)           
			batch_z = np.random.normal(0,1,[self.batch_size, z_dim]).astype(np.float32)
			return batch[0].reshape([self.batch_size, 28,28,1])* 255.0, batch_z

	def train(self):
		log_dir = 'BEgan'+'unet'
		
		self.D_train = tf.train.RMSPropOptimizer(learning_rate = 1e-3).minimize(self.D_loss, var_list = self.theta_c)
		self.G_train = tf.train.RMSPropOptimizer(learning_rate = 1e-3).minimize(self.G_loss, var_list = self.theta_g)
		
		tf.add_to_collection('z', self.z)
		# tf.add_to_collection('L_img', self.images)
		tf.add_to_collection('fake_img', self.fake)
		tf.add_to_collection('H_img', self.labels)

		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter('./log/'+log_dir, self.sess.graph)
		tf.global_variables_initializer().run()
		start_time = time.time()

		
		
		for counter in xrange(100000):
			citers = 5
			if (counter < 25) or (counter % 100 == 0):
				citers = 50
			# pt += citers*counter*self.batch_size//len(train_data)		
			for i in xrange(citers):
				label, z = self.next_feed_dict()
				summary,_,_ = self.sess.run([merged,self.D_train,self.kt_change],feed_dict = {self.labels: label, self.z: z})
				train_writer.add_summary(summary, counter*50+i)

			label, z = self.next_feed_dict()
			c_0,c0,label0,fake0,_, g_loss, measure, _, d_loss, kt, kt_=self.sess.run([self.c_0,self.c0,self.label0,self.fake0,self.G_train, self.G_loss, self.measure, self.kt_change, self.D_loss, self.kt, self.kt_], feed_dict = {self.labels: label, self.z: z})
			

			if counter % 30 == 0:
				print("step: [%2d], time: [%4.4f], measure: [%.5f]  g_loss: [%.5f],  d_loss:  [%.5f],   kt:   [%.5f]  kt_:  [%.5f]\n" %(counter, time.time()-start_time,\
				 measure, g_loss, d_loss, kt, kt_))

			if counter % 15 == 0:
				size = 28
				
				pics = self.sess.run(self.fake, feed_dict = {self.z: tz})
				
				pic = np.zeros([size*28, size, 1])
				
				pic[size:size*2, :, :] = label0/255.0
				
				pic[size*2:size*3, :, :] = fake0/255.0

				pic[size*3:size*4,:,:] = c_0/255.0
				
				pic[size*4:size*5,:,:] = c0/255.0

				for i in range(20):
					pic[size*(6+i):size*(7+i),:,:] = pics[i]/255.0




				cv2.imshow('img', pic)
				cv2.waitKey(1)
			if counter % 3000 == 0:
				pic = (pic*255).astype(np.int32)
				cv2.imwrite('./pictures/img'+str(counter)+'.jpg', pic)



			if counter % 10000 == 0:
				self.save_dir = 'ckpt_began/'+'counter'+str(counter)+'began'+'mnist.ckpt'
				self.saver.save(self.sess, self.save_dir)
		
	
	