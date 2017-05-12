import numpy as np
import scipy.misc
import os
import glob
data_dir = '/media/shj/0BEE066D0BEE066D/DataSets/CelebA/Img/img_align_celeba'

def prepare_datalist():
	# filenames = os.listdir(dataset)
	dir_list = glob.glob(os.path.join(data_dir, '*.jpg'))
	#return a list contains all the absolute path of every bmp file
	return dir_list

def imread(img_dir):
	img = scipy.misc.imread(img_dir, mode = 'RGB').astype(np.float)
	return img

class data_reader(object):
	def __init__(self, batch_size):
		self.size = batch_size
		self.pt = 0	
		self.dir_list = prepare_datalist()

	def next_batch(self):
		imgs = []
		for i in range(self.size):
			img = imread(self.dir_list[self.pt+i])
			imgs.append(scipy.misc.imresize(img[:, 20:198,:],(64,64,3)).reshape([64,64,3]))
		self.pt+=self.size
		if self.pt>=len(self.dir_list)or(self.pt+self.size > len(self.dir_list)):
			self.pt = 0
		return np.stack(imgs,axis = 0)


def lrelu(x, leak = 0.3, name = 'lrelu'):
	f1 = 0.5*(1+leak)
	f2 = 0.5*(1-leak)
	return f1*x+f2*abs(x)

def bgr2rgb(img):
	assert (len(img.shape) == 3)
	out = np.zeros(img.shape)
	out[:,:,0] = img[:,:,2]
	out[:,:,1] = img[:,:,1]
	out[:,:,2] = img[:,:,0]

	return out