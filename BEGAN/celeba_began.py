import time
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl
import math
import utils
# import scipy.misc

from tensorflow.examples.tutorials.mnist import input_data



batch_size = 32
z_dim = 128

data = utils.data_reader(batch_size)


def lrelu(x, leak = 0.3, name = 'lrelu'):
	f1 = 0.5*(1+leak)
	f2 = 0.5*(1-leak)
	return f1*x+f2*abs(x)


def Generator(z):
	with tf.variable_scope('gen') as scope:
	    train = tcl.fully_connected(
	        z, 4 * 4 * 128, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
	    train = tf.reshape(train, (-1, 4, 4, 128))
	    train = tcl.conv2d_transpose(train, 128, 3, stride=2,	#4-8
	                                activation_fn=lrelu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
	    train = tcl.conv2d_transpose(train, 128, 3, stride=2,	#8-16
	                                activation_fn=lrelu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
	    train = tcl.conv2d_transpose(train, 64, 3, stride=2,	#16-32
	                                activation_fn=lrelu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
	    train = tcl.conv2d_transpose(train, 64, 3, stride=2,	#32-64
	                                activation_fn=lrelu, normalizer_fn=tcl.batch_norm,  padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
	    train = tcl.conv2d(train, num_outputs = 3, stride = 1, kernel_size = 3,
	    	activation_fn = tf.nn.tanh, padding = 'SAME')
	    train = (train+1.0)/2.0 * 255.0
	    return train

def AE(H_img,reuse = False):
	
	# L_img = tf.reshape(L_img, [batch_size, 64, 64, 3])
	with tf.variable_scope('crit') as scope:
		if reuse:
			scope.reuse_variables()

		# conv1 = tcl.conv2d(H_img, num_outputs = 64, stride = 2, kernel_size = 3,		#128->64 
		# 	activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding = 'SAME')

		conv2 = tcl.conv2d(H_img, num_outputs = 64, stride = 2, kernel_size = 3,		#64->32	64
			activation_fn = tf.nn.relu, normalizer_fn = tcl.batch_norm, padding = 'SAME')

		conv3 = tcl.conv2d(conv2, num_outputs = 64, stride = 2, kernel_size = 3,		#32->16	64
			activation_fn = tf.nn.relu, normalizer_fn = tcl.batch_norm, padding = 'SAME')

		conv4 = tcl.conv2d(conv3, num_outputs = 128, stride = 2, kernel_size = 3,		#16->8	128
			activation_fn = tf.nn.relu, normalizer_fn = tcl.batch_norm, padding = 'SAME')

		conv5 = tcl.conv2d(conv4, num_outputs = 128, stride = 2, kernel_size = 3,		#8->4	128
			activation_fn = tf.nn.relu, normalizer_fn = tcl.batch_norm, padding = 'SAME')

		# conv6 = tcl.conv2d(conv5, num_outputs = 128, stride = 2, kernel_size = 3,		#4->2	128
		# 	activation_fn = tf.nn.relu, normalizer_fn = tcl.batch_norm, padding = 'SAME')

		pre = tcl.flatten(conv5)

		z = tcl.fully_connected(pre, num_outputs = 128, activation_fn = lrelu)

		post = tcl.fully_connected(z, num_outputs = 4*4*128, activation_fn = lrelu)

		post = tf.reshape(post, [batch_size, 4, 4, 128])							#4 256

		deconv1 = tcl.conv2d_transpose(post, 128, kernel_size = 3, stride = 2, 		# 4->8   128
			activation_fn=lrelu, normalizer_fn=tcl.batch_norm, 
	        padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
		deconv2 = tcl.conv2d_transpose(deconv1, 64, kernel_size = 3, stride = 2, 		# 8->16	 64
			activation_fn=lrelu, normalizer_fn=tcl.batch_norm, 
	        padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
		deconv3 = tcl.conv2d_transpose(deconv2, 64, kernel_size = 3, stride = 2, 		# 16->32  64
			activation_fn=lrelu, normalizer_fn=tcl.batch_norm, 
	        padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
		deconv4 = tcl.conv2d_transpose(deconv3, 64, kernel_size = 3, stride = 2, 		#32->64  64
			activation_fn=lrelu, normalizer_fn=tcl.batch_norm,
	        padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
		
		img = tcl.conv2d(deconv4, num_outputs = 3, stride = 1, kernel_size = 3,
	    	activation_fn = tf.nn.tanh, padding = 'SAME')


	return (img+1.0)/2.0 * 255.0

 


class BEgan(object):

	def __init__(self, sess,
		image_size = 64,
		label_size = 64,
		batch_size = batch_size,
		channels = 3,
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
			batch = data.next_batch()           
			batch_z = np.random.normal(0,1,[self.batch_size, z_dim]).astype(np.float32)
			return batch,  batch_z

	def train(self):
		log_dir = 'BEgan'+'unet'
		# with tf.variable_scope('train'):
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
				citers = 20
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
				size = self.image_size
				
				# pics = self.sess.run(self.fake, feed_dict = {self.z: tz})
				
				pic = np.zeros([size*6, size, 3])
				# pic[size:size*2, :, :] = label0[:,:,:]/255.0
				pic[size:size*2, :, :] = utils.bgr2rgb(label0)/255.0

				# pic[size:size*2, :, 0] = label0[:,:,2]/255.0
				# pic[size:size*2, :, 1] = label0[:,:,1]/255.0
				# pic[size:size*2, :, 2] = label0[:,:,0]/255.0

				pic[size*2:size*3, :, :] = utils.bgr2rgb(fake0)/255.0
				# pic[size*2:size*3, :, 0] = fake0[:,:,2]/255.0
				# pic[size*2:size*3, :, 1] = fake0[:,:,1]/255.0
				# pic[size*2:size*3, :, 2] = fake0[:,:,0]/255.0

				pic[size*3:size*4,:,:] = utils.bgr2rgb(c_0)/255.0
				# pic[size*3:size*4,:,0] = c_0[:,:,2]/255.0
				# pic[size*3:size*4,:,1] = c_0[:,:,1]/255.0
				# pic[size*3:size*4,:,2] = c_0[:,:,0]/255.0
				
				pic[size*4:size*5,:,:] = utils.bgr2rgb(c0)/255.0
				# pic[size*4:size*5,:,0] = c0[:,:,2]/255.0
				# pic[size*4:size*5,:,1] = c0[:,:,1]/255.0
				# pic[size*4:size*5,:,2] = c0[:,:,0]/255.0

				cv2.imshow('img', pic)
				cv2.waitKey(1)

			if counter % 3000 == 0:
				pic = (pic*255).astype(np.int32)
				cv2.imwrite('./pictures/img'+str(counter)+'.jpg', pic)



			if counter % 10000 == 0:
				self.save_dir = 'ckpt_began/'+'counter'+str(counter)+'began'+'celeba.ckpt'
				# self.restore_dir = self.save_dir
				self.saver.save(self.sess, self.save_dir)
		
	
	def test(self):
		self.restore_dir = 'ckpt_began/counter60000beganmnist.ckpt'
		self.saver.restore(self.sess, self.restore_dir)
		test_size = 20
		test_z = np.random.normal(0,1,[batch_size, z_dim]).astype(np.float32)
		test_fake, test_ae_ = self.sess.run([self.fake, self.AE_], feed_dict={self.z : test_z})
		pic = np.zeros([128, test_size*64, 3])
		for i in range(test_size):
			pic[:64, i*64:(i+1)*64,:] = utils.bgr2rgb(test_fake[i,:,:,:])/255.0
			pic[64:128, i*64:(i+1)*64,:] = utils.bgr2rgb(test_ae_[i,:,:,:])/255.0

		cv2.imshow('test_img', pic)
		cv2.waitKey(0)
		pic = (pic*255).astype(np.int32)
		cv2.imwrite('./pictures/60000img'+'3.jpg', pic)
